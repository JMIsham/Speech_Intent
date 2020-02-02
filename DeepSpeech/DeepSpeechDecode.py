import os
import sys

import progressbar

import numpy as np
import pandas as pd
import tensorflow as tf

from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from ds_util.audio import audio_file_to_input_vector
from ds_util.config import Config, initialize_globals
from ds_util.flags import create_flags, FLAGS
from ds_util.logging import log_info, log_error, log_debug, log_warn

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = \
    sys.argv[log_level_index] if log_level_index and log_level_index < len(sys.argv) else '3'


def variable_on_worker_level(name, shape, initializer):
    r'''
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_worker_level()``
    used to create a variable in CPU memory.
    '''
    # Use the /cpu:0 device on worker_device for scoped operations
    if len(FLAGS.ps_hosts) == 0:
        device = Config.worker_device
    else:
        device = tf.train.replica_device_setter(worker_device=Config.worker_device, cluster=Config.cluster)

    with tf.device(device):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def BiRNN(batch_x, seq_length, dropout, reuse=False, batch_size=None, n_steps=-1, previous_state=None, tflite=False):
    r'''
    That done, we will define the learned variables, the weights and biases,
    within the method ``BiRNN()`` which also constructs the neural network.
    The variables named ``hn``, where ``n`` is an integer, hold the learned weight variables.
    The variables named ``bn``, where ``n`` is an integer, hold the learned bias variables.
    In particular, the first variable ``h1`` holds the learned weight matrix that
    converts an input vector of dimension ``n_input + 2*n_input*n_context``
    to a vector of dimension ``n_hidden_1``.
    Similarly, the second variable ``h2`` holds the weight matrix converting
    an input vector of dimension ``n_hidden_1`` to one of dimension ``n_hidden_2``.
    The variables ``h3``, ``h5``, and ``h6`` are similar.
    Likewise, the biases, ``b1``, ``b2``..., hold the biases for the various layers.
    '''
    layers = {}

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    if not batch_size:
        batch_size = tf.shape(batch_x)[0]

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2, 3])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, Config.n_input + 2*Config.n_input*Config.n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)
    layers['input_reshaped'] = batch_x

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    b1 = variable_on_worker_level('b1', [Config.n_hidden_1], tf.zeros_initializer())
    h1 = variable_on_worker_level('h1', [Config.n_input + 2*Config.n_input*Config.n_context, Config.n_hidden_1], tf.contrib.layers.xavier_initializer())
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), FLAGS.relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))
    layers['layer_1'] = layer_1

    # 2nd layer
    b2 = variable_on_worker_level('b2', [Config.n_hidden_2], tf.zeros_initializer())
    h2 = variable_on_worker_level('h2', [Config.n_hidden_1, Config.n_hidden_2], tf.contrib.layers.xavier_initializer())
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), FLAGS.relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))
    layers['layer_2'] = layer_2

    # 3rd layer
    b3 = variable_on_worker_level('b3', [Config.n_hidden_3], tf.zeros_initializer())
    h3 = variable_on_worker_level('h3', [Config.n_hidden_2, Config.n_hidden_3], tf.contrib.layers.xavier_initializer())
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), FLAGS.relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))
    layers['layer_3'] = layer_3

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell:
    if not tflite:
        fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(Config.n_cell_dim, reuse=reuse)
        layers['fw_cell'] = fw_cell
    else:
        fw_cell = tf.nn.rnn_cell.LSTMCell(Config.n_cell_dim, reuse=reuse)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM RNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [n_steps, batch_size, Config.n_hidden_3])
    if tflite:
        # Generated StridedSlice, not supported by NNAPI
        #n_layer_3 = []
        #for l in range(layer_3.shape[0]):
        #    n_layer_3.append(layer_3[l])
        #layer_3 = n_layer_3

        # Unstack/Unpack is not supported by NNAPI
        layer_3 = tf.unstack(layer_3, n_steps)

    # We parametrize the RNN implementation as the training and inference graph
    # need to do different things here.
    if not tflite:
        output, output_state = fw_cell(inputs=layer_3, dtype=tf.float32, sequence_length=seq_length, initial_state=previous_state)
    else:
        output, output_state = tf.nn.static_rnn(fw_cell, layer_3, previous_state, tf.float32)
        output = tf.concat(output, 0)

    # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
    # to a tensor of shape [n_steps*batch_size, n_cell_dim]
    output = tf.reshape(output, [-1, Config.n_cell_dim])
    layers['rnn_output'] = output
    layers['rnn_output_state'] = output_state

    # Now we feed `output` to the fifth hidden layer with clipped RELU activation and dropout
    b5 = variable_on_worker_level('b5', [Config.n_hidden_5], tf.zeros_initializer())
    h5 = variable_on_worker_level('h5', [Config.n_cell_dim, Config.n_hidden_5], tf.contrib.layers.xavier_initializer())
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(output, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))
    layers['layer_5'] = layer_5

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_worker_level('b6', [Config.n_hidden_6], tf.zeros_initializer())
    h6 = variable_on_worker_level('h6', [Config.n_hidden_5, Config.n_hidden_6], tf.contrib.layers.xavier_initializer())
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
    layers['layer_6'] = layer_6

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [n_steps, batch_size, Config.n_hidden_6], name="raw_logits")
    layers['raw_logits'] = layer_6

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, layers


def create_inference_graph(batch_size=1, n_steps=16, tflite=False):
    # Input tensor will be of shape [batch_size, n_steps, 2*n_context+1, n_input]
    input_tensor = tf.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2 * Config.n_context + 1,
                                               Config.n_input], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    if not tflite:
        previous_state_c = variable_on_worker_level('previous_state_c', [batch_size, Config.n_cell_dim],
                                                    initializer=None)
        previous_state_h = variable_on_worker_level('previous_state_h', [batch_size, Config.n_cell_dim],
                                                    initializer=None)
    else:
        previous_state_c = tf.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_c')
        previous_state_h = tf.placeholder(tf.float32, [batch_size, Config.n_cell_dim], name='previous_state_h')

    previous_state = tf.contrib.rnn.LSTMStateTuple(previous_state_c, previous_state_h)

    no_dropout = [0.0] * 6

    logits, layers = BiRNN(batch_x=input_tensor,
                           seq_length=seq_length if FLAGS.use_seq_length else None,
                           dropout=no_dropout,
                           batch_size=batch_size,
                           n_steps=n_steps,
                           previous_state=previous_state,
                           tflite=tflite)

    # TF Lite runtime will check that input dimensions are 1, 2 or 4
    # by default we get 3, the middle one being batch_size which is forced to
    # one on inference graph, so remove n_that dimension
    if tflite:
        logits = tf.squeeze(logits, [1])

    # Apply softmax for CTC decoder
    logits = tf.nn.softmax(logits)

    new_state_c, new_state_h = layers['rnn_output_state']

    # Initial zero state
    if not tflite:
        zero_state = tf.zeros([batch_size, Config.n_cell_dim], tf.float32)
        initialize_c = tf.assign(previous_state_c, zero_state)
        initialize_h = tf.assign(previous_state_h, zero_state)
        initialize_state = tf.group(initialize_c, initialize_h, name='initialize_state')
        with tf.control_dependencies(
                [tf.assign(previous_state_c, new_state_c), tf.assign(previous_state_h, new_state_h)]):
            logits = tf.identity(logits, name='logits')

        return (
            {
                'input': input_tensor,
                'input_lengths': seq_length,
            },
            {
                'outputs': logits,
                'initialize_state': initialize_state,
            },
            layers
        )
    else:
        logits = tf.identity(logits, name='logits')
        new_state_c = tf.identity(new_state_c, name='new_state_c')
        new_state_h = tf.identity(new_state_h, name='new_state_h')

        return (
            {
                'input': input_tensor,
                'previous_state_c': previous_state_c,
                'previous_state_h': previous_state_h,
            },
            {
                'outputs': logits,
                'new_state_c': new_state_c,
                'new_state_h': new_state_h,
            },
            layers
        )


def do_single_file_inference(input_file_path):
    with tf.Session(config=Config.session_config) as session:
        inputs, outputs, _ = create_inference_graph(batch_size=1, n_steps=-1)

        # Create a saver using variables from the above newly created graph
        mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        saver = tf.train.Saver(mapping)

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counteract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error(
                'Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        session.run(outputs['initialize_state'])

        features = audio_file_to_input_vector(input_file_path, Config.n_input, Config.n_context)
        num_strides = len(features) - (Config.n_context * 2)

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2 * Config.n_context + 1
        features = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, Config.n_input),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)

        logits = session.run(outputs['outputs'], feed_dict={
            inputs['input']: [features],
            inputs['input_lengths']: [num_strides],
        })

        logits = np.squeeze(logits)

        # scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
        #                 FLAGS.lm_binary_path, FLAGS.lm_trie_path,
        #                 Config.alphabet)

        decoded = ctc_beam_search_decoder(logits, Config.alphabet, FLAGS.beam_width)
        # Print highest probability result
        print(decoded[0][1])


def batch_inference(file_names):

    # constants
    c_max_rows = 555

    create_flags()
    initialize_globals()

    output_features = []

    with tf.Session(config=Config.session_config) as session:
        inputs, outputs, _ = create_inference_graph(batch_size=1, n_steps=-1)

        # Create a saver using variables from the above newly created graph
        mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        saver = tf.train.Saver(mapping)

        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error(
                'Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path

        bar = progressbar.ProgressBar()

        for file in bar(file_names):

            saver.restore(session, checkpoint_path)
            session.run(outputs['initialize_state'])

            features = audio_file_to_input_vector(file, Config.n_input, Config.n_context)
            num_strides = len(features) - (Config.n_context * 2)

            # Create a view into the array with overlapping strides of size
            # numcontext (past) + 1 (present) + numcontext (future)
            window_size = 2 * Config.n_context + 1
            features = np.lib.stride_tricks.as_strided(
                features,
                (num_strides, window_size, Config.n_input),
                (features.strides[0], features.strides[0], features.strides[1]),
                writeable=False)

            logits = session.run(outputs['outputs'], feed_dict={
                inputs['input']: [features],
                inputs['input_lengths']: [num_strides],
            })

            logits = np.squeeze(logits)
            # decoded = ctc_beam_search_decoder(logits, Config.alphabet, FLAGS.beam_width)
            # output_features.append(decoded[0][1])
            logits = np.pad(logits, ((0, c_max_rows - logits.shape[0]), (0, 0)), 'constant')
            output_features.append(logits)

    return output_features


def main(_):
    # audio folder
    audio_folder = 'data/tamil_2/'

    # read csv from csv
    # data = pd.read_csv('data/tamil_data_v2.csv')
    # file_names = data['n_path']
    # file_names = file_names.apply(lambda x: audio_folder + str(x) + '.wav')
    # file_names.values

    file_names = np.array([
        'data/formatted_data/audio1527241893196_3_4.wav',
        'data/formatted_data/audio1527227206340_5_3.wav',
        'data/formatted_data/audio1527224098833_6_4.wav',
        'data/formatted_data/audio1527158207821_1_7.wav'
    ])

    output = batch_inference(file_names)

    # max_len = max(map(len, output))
    # output = np.array([np.pad(v, [0, max_len - len(v)], 'constant') for v in output])
    # data = np.c_[data['filename'], data['category_1'], data['category_2'], data['length'], output]
    # data = pd.DataFrame(data)
    # data.to_csv('data/ds_decode_data_v1.csv', index=False)

    # save np array to file with .npy
    output = np.array(output)
    # np.save('data/ds_decode_tamil data_v2_padded', output)


if __name__ == '__main__':
    # create_flags()
    tf.app.run(main)
