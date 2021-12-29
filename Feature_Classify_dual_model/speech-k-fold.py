# K fol cross validation for speech data

# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from datetime import datetime

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn import svm
import tensorflow

# -----------------------------------------------------------------------------

lang = 'ta'  # 'si', 'ta'
expr = 'wav2vec'  # 'ds1', 'ds2', 'phonemes', wav2vec
mtype = '2d_cnn'  # '1d_cnn', '2d_cnn', 'svm'
merger = 'ds2'  # 'ds1', 'ds2', 'phonemes', wav2vec


# random number
random_seed = 10
np.random.seed(random_seed)
# set_random_seed(random_seed)
tensorflow.random.set_seed(random_seed)


def get_data(lang, expr, mtype):
    print('Running on ', lang, ' data')

    csv_file = {
        'si': '../input/speechsi/formatted_data_v2.csv',
        'ta': '../data/Tamil_Dataset/Tamil_Data.csv'
    }
    datax_dic = {
        'si': {
            'ds1': '../input/speechsi/ds_decode_data_v2_padded.npy',
            'ds2': '../input/speechds2/ds2_decode_sinhala_data_v2_padded.npy',
            'phonemes': '../input/speech-phonemes/phoneme_decode_sinhala_data_v2_padded.npy'
        },
        'ta': {
            'ds1': '../input/speechta/ds_decode_tamil_data_v2_padded.npy',
            'ds2': '../data/Tamil_Dataset/ds2_decode_Tamil_data_v2.npy',
            'phonemes': '../data/Tamil_Dataset/phoneme_decode_Tamil_data_v2.npy',
            'wav2vec': '../data/Tamil_Dataset/wav2vec_decode_Tamil_data_v2.npy'

    }
    }

    # read csv and get classes
    data = pd.read_csv(csv_file[lang])
    data_y = data['intent']

    # load ds decoded output
    data_x = np.load(datax_dic[lang][expr])
    data_x_merge = np.load(datax_dic[lang][merger], allow_pickle=True)
    data_x = np.concatenate((data_x, data_x_merge), axis=1)

    # reshape data for CNN
    # data_x = np.reshape(data_x, (data_x.shape[0], 555, 29, 1))
    if mtype == '1d_cnn':
        data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], data_x.shape[2]))
    if mtype == '2d_cnn':
        data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], data_x.shape[2], 1))
    if mtype == 'svm':
        data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1] * data_x.shape[2]))

    num_of_classes = len(data_y.unique())
    print('Total data samples   :', data_x.shape[0])
    print('Number of classes    :', num_of_classes)

    # one hot encoding y values
    data_y = pd.get_dummies(data_y)
    data_y = data_y.values

    # calculate total occurrences of classes
    print('Occurrences of classes in total dataset')
    print(data_y[:, 0:6].sum(axis=0))

    # print data set sizes
    print()
    print('----Dataset Detail----')
    print('Training Data X  :', data_x.shape, ' Y', data_y.shape)

    print('--Occurrences in each classes--')
    print('Training Data   :', data_y.sum(axis=0))

    return data_x, data_y


def get_mfcc_data(dataset):
    print('Running on ', dataset, 'mfcc data')

    csv_file = {
        'si': '../input/speechsi/formatted_data_v2.csv',
        'ta': '../input/speechta/tamil_data_v2.csv'
    }
    feature_file = {
        'si': '../input/speechmfcc/mfcc_si_data_v2_padded.npy',
        'ta': '../input/speechmfcc/mfcc_ta_data_v2_padded.npy'
    }

    # read csv and get classes
    data = pd.read_csv(csv_file[dataset])
    data_y = data['category_1']

    # load ds decoded output
    data_x = np.load(feature_file[dataset])

    # reshape data
    data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1] * data_x.shape[2]))

    num_of_classes = len(data_y.unique())
    print('Total data samples   :', data_x.shape[0])
    print('Number of classes    :', num_of_classes)

    # one hot encoding y values
    data_y = pd.get_dummies(data_y)
    data_y = data_y.values

    # calculate total occurrences of classes
    print('Occurrences of classes in total dataset')
    print(data_y[:, 0:6].sum(axis=0))

    # print data set sizes
    print()
    print('----Dataset Detail----')
    print('Training Data X  :', data_x.shape, ' Y', data_y.shape)

    print('--Occurrences in each classes--')
    print('Training Data   :', data_y.sum(axis=0))

    return data_x, data_y


# ------------------------------------------------------------------
# models

def get_cnn1d_model1(param, num_of_classes, input_shape):
    model_ = Sequential()

    model_.add(Conv1D(
        param['m_filters1'],
        param['m_kernel1_size'],
        activation='relu',
        padding='same',
        # input_shape=(256, 42)
        # input_shape=(555, 29)
        input_shape=input_shape
    ))
    model_.add(MaxPooling1D(
        pool_size=param['m_pool1_size'],
        strides=param['m_stride1_size']
    ))
    model_.add(Dropout(
        param['m_dropout']
    ))
    model_.add(Conv1D(
        param['m_filters2'],
        param['m_kernel2_size'],
        activation='relu',
        padding='same'
    ))
    model_.add(MaxPooling1D(
        pool_size=param['m_pool2_size'],
        strides=param['m_stride2_size'],
    ))
    model_.add(Dropout(
        param['m_dropout']
    ))
    model_.add(Flatten())
    model_.add(Dense(
        param['m_hidden_units'],
        activation='relu'
    ))
    model_.add(Dense(
        num_of_classes,
        activation='softmax'
    ))

    # Compile model
    model_.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

    return model_


def get_model_1d_cnn(dataset, input_shape):
    parameters1d_si = {
        'f_batch_size': 3,
        'm_dropout': 0.1,
        'm_filters1': 38,
        'm_filters2': 28,
        'm_hidden_units': 131,
        'm_kernel1_size': 19,
        'm_kernel2_size': 22,
        'm_pool1_size': 18,
        'm_pool2_size': 22,
        'm_stride1_size': 7,
        'm_stride2_size': 10
    }

    # parameters1d_ta = {
    #     'f_batch_size': 3,
    #     'm_dropout': 0.1,
    #     'm_filters1': 39,
    #     'm_filters2': 26,
    #     'm_hidden_units': 84,
    #     'm_kernel1_size': 18,
    #     'm_kernel2_size': 19,
    #     'm_pool1_size': 25,
    #     'm_pool2_size': 20,
    #     'm_stride1_size': 5,
    #     'm_stride2_size': 5
    # }
    parameters1d_ta = {
        'f_batch_size': 1.0,
        'm_dropout': 0.05,
        'm_filters1': 43.0,
        'm_filters2': 17.0,
        'm_hidden_units': 46.0,
        'm_kernel1_size': 31.0,
        'm_kernel2_size': 17.0,
        'm_pool1_size': 47.0,
        'm_pool2_size': 41.0,
        'm_stride1_size': 2.0,
        'm_stride2_size': 27.0
    }

    print('Building model for ', dataset, ' dataset')
    if dataset == 'si':
        model_ = get_cnn1d_model1(parameters1d_si, 6, input_shape)
    else:
        model_ = get_cnn1d_model1(parameters1d_ta, 6, input_shape)

    return model_


def get_model_1d_cnn_ds2(dataset, input_shape):
    parameters1d_si = {
        'f_batch_size': 2,
        'm_dropout': 0.05,
        'm_filters1': 46,
        'm_filters2': 44,
        'm_hidden_units': 160,
        'm_kernel1_size': 9,
        'm_kernel2_size': 34,
        'm_pool1_size': 17,
        'm_pool2_size': 47,
        'm_stride1_size': 8,
        'm_stride2_size': 17
    }

    # parameters1d_ta = {
    #     'f_batch_size': 4,
    #     'm_dropout': 0.1,
    #     'm_filters1': 42,
    #     'm_filters2': 40,
    #     'm_hidden_units': 115,
    #     'm_kernel1_size': 21,
    #     'm_kernel2_size': 27,
    #     'm_pool1_size': 47,
    #     'm_pool2_size': 24,
    #     'm_stride1_size': 4,
    #     'm_stride2_size': 18
    # }

    parameters1d_ta = {'f_batch_size': 10,
                       'm_dropout': 0.1,
                       'm_filters1': 29,
                       'm_filters2': 32,
                       'm_hidden_units': 80,
                       'm_kernel1_size': 3,
                       'm_kernel2_size': 29,
                       'm_pool1_size': 33,
                       'm_pool2_size': 49,
                       'm_stride1_size': 9,
                       'm_stride2_size': 37}


    print('Building model for ', dataset, ' dataset')
    if dataset == 'si':
        model_ = get_cnn1d_model1(parameters1d_si, 6, input_shape)
    else:
        model_ = get_cnn1d_model1(parameters1d_ta, 6, input_shape)

    return model_


def get_model_1d_cnn_pho(dataset, input_shape):
    parameters1d_si = {
        'f_batch_size': 4,
        'm_dropout': 0.2,
        'm_filters1': 47,
        'm_filters2': 41,
        'm_hidden_units': 123,
        'm_kernel1_size': 37,
        'm_kernel2_size': 35,
        'm_pool1_size': 28,
        'm_pool2_size': 24,
        'm_stride1_size': 1,
        'm_stride2_size': 11
    }

    # parameters1d_ta = {
    #     'f_batch_size': 4,
    #     'm_dropout': 0.3,
    #     'm_filters1': 11,
    #     'm_filters2': 37,
    #     'm_hidden_units': 71,
    #     'm_kernel1_size': 18,
    #     'm_kernel2_size': 17,
    #     'm_pool1_size': 18,
    #     'm_pool2_size': 35,
    #     'm_stride1_size': 3,
    #     'm_stride2_size': 1
    # }
    parameters1d_ta = {
        'f_batch_size': 2,
        'm_dropout': 0.0,
        'm_filters1': 28,
        'm_filters2': 43,
        'm_hidden_units': 91,
        'm_kernel1_size': 28,
        'm_kernel2_size': 30,
        'm_pool1_size': 11,
        'm_pool2_size': 29,
        'm_stride1_size': 2,
        'm_stride2_size': 33}


    print('Building model for ', dataset, ' dataset')
    if dataset == 'si':
        model_ = get_cnn1d_model1(parameters1d_si, 6, input_shape)
    else:
        model_ = get_cnn1d_model1(parameters1d_ta, 6, input_shape)

    return model_

def get_model_1d_cnn_wav2vec(dataset, input_shape):
    parameters1d_si = {
        'f_batch_size': 4,
        'm_dropout': 0.2,
        'm_filters1': 47,
        'm_filters2': 41,
        'm_hidden_units': 123,
        'm_kernel1_size': 37,
        'm_kernel2_size': 35,
        'm_pool1_size': 28,
        'm_pool2_size': 24,
        'm_stride1_size': 1,
        'm_stride2_size': 11
    }

    # parameters1d_ta = {
    #     'f_batch_size': 4,
    #     'm_dropout': 0.3,
    #     'm_filters1': 11,
    #     'm_filters2': 37,
    #     'm_hidden_units': 71,
    #     'm_kernel1_size': 18,
    #     'm_kernel2_size': 17,
    #     'm_pool1_size': 18,
    #     'm_pool2_size': 35,
    #     'm_stride1_size': 3,
    #     'm_stride2_size': 1
    # }
    parameters1d_ta = {
        'f_batch_size': 2,
        'm_dropout': 0.0,
        'm_filters1': 28,
        'm_filters2': 43,
        'm_hidden_units': 91,
        'm_kernel1_size': 28,
        'm_kernel2_size': 30,
        'm_pool1_size': 11,
        'm_pool2_size': 29,
        'm_stride1_size': 2,
        'm_stride2_size': 33}


    print('Building model for ', dataset, ' dataset')
    if dataset == 'si':
        model_ = get_cnn1d_model1(parameters1d_si, 6, input_shape)
    else:
        model_ = get_cnn1d_model1(parameters1d_ta, 6, input_shape)

    return model_


def get_cnn_2d_model3(param, num_of_classes, input_shape):
    model_ = Sequential()

    model_.add(Conv2D(
        param['m_filters1'],
        (param['m_kernel1_size_x'], param['m_kernel1_size_y']),
        activation='relu',
        padding='same',
        # input_shape=(555, 29, 1)
        input_shape=input_shape
    ))
    model_.add(MaxPooling2D(
        pool_size=(param['m_pool1_size_x'], param['m_pool1_size_y']),
        strides=(param['m_stride1_size_x'], param['m_stride1_size_y'])
    ))
    model_.add(Dropout(
        param['m_dropout']
    ))
    model_.add(Conv2D(
        param['m_filters2'],
        (param['m_kernel2_size_x'], param['m_kernel2_size_y']),
        activation='relu',
        padding='same'
    ))
    model_.add(MaxPooling2D(
        pool_size=(param['m_pool2_size_x'], param['m_pool2_size_y']),
        strides=(param['m_stride2_size_x'], param['m_stride2_size_y'])
    ))
    model_.add(Dropout(
        param['m_dropout']
    ))
    model_.add(Flatten())
    model_.add(Dense(
        param['m_hidden_units'],
        activation='relu'
    ))
    model_.add(Dense(
        num_of_classes,
        activation='softmax'
    ))

    # Compile model
    model_.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

    return model_


def get_model_2d_cnn(dataset, input_shape):
    parameters2d_si = {
        'f_batch_size': 5,
        'm_dropout': 0.1,
        'm_filters1': 16,
        'm_filters2': 17,
        'm_hidden_units': 118,
        'm_kernel1_size_x': 1,
        'm_kernel1_size_y': 8,
        'm_kernel2_size_x': 20,
        'm_kernel2_size_y': 8,
        'm_pool1_size_x': 6,
        'm_pool1_size_y': 1,
        'm_pool2_size_x': 19,
        'm_pool2_size_y': 2,
        'm_stride1_size_x': 5,
        'm_stride1_size_y': 5,
        'm_stride2_size_x': 16,
        'm_stride2_size_y': 8
    }

    parameters2d_ta = {
        'f_batch_size': 1,
        'm_dropout': 0.3,
        'm_filters1': 14,
        'm_filters2': 13,
        'm_hidden_units': 127,
        'm_kernel1_size_x': 5,
        'm_kernel1_size_y': 1,
        'm_kernel2_size_x': 11,
        'm_kernel2_size_y': 20,
        'm_pool1_size_x': 13,
        'm_pool1_size_y': 1,
        'm_pool2_size_x': 17,
        'm_pool2_size_y': 1,
        'm_stride1_size_x': 5,
        'm_stride1_size_y': 1,
        'm_stride2_size_x': 2,
        'm_stride2_size_y': 7
    }

    print('Building model for ', dataset, ' dataset')
    if dataset == 'si':
        model_ = get_cnn_2d_model3(parameters2d_si, 6, input_shape)
    else:
        model_ = get_cnn_2d_model3(parameters2d_ta, 6, input_shape)

    return model_


def get_model_2d_cnn_ds2(dataset, input_shape):
    parameters2d_si = {
        'f_batch_size': 4,
        'm_dropout': 0.15,
        'm_filters1': 19,
        'm_filters2': 15,
        'm_hidden_units': 128,
        'm_kernel1_size_x': 4,
        'm_kernel1_size_y': 5,
        'm_kernel2_size_x': 14,
        'm_kernel2_size_y': 19,
        'm_pool1_size_x': 10,
        'm_pool1_size_y': 1,
        'm_pool2_size_x': 15,
        'm_pool2_size_y': 1,
        'm_stride1_size_x': 6,
        'm_stride1_size_y': 2,
        'm_stride2_size_x': 3,
        'm_stride2_size_y': 5
    }

    parameters2d_ta = {
        'f_batch_size': 2,
        'm_dropout': 0.1,
        'm_filters1': 20,
        'm_filters2': 7,
        'm_hidden_units': 130,
        'm_kernel1_size_x': 6,
        'm_kernel1_size_y': 1,
        'm_kernel2_size_x': 20,
        'm_kernel2_size_y': 17,
        'm_pool1_size_x': 10,
        'm_pool1_size_y': 1,
        'm_pool2_size_x': 16,
        'm_pool2_size_y': 1,
        'm_stride1_size_x': 6,
        'm_stride1_size_y': 1,
        'm_stride2_size_x': 2,
        'm_stride2_size_y': 5
    }

    print('Building model for ', dataset, ' dataset')
    if dataset == 'si':
        model_ = get_cnn_2d_model3(parameters2d_si, 6, input_shape)
    else:
        model_ = get_cnn_2d_model3(parameters2d_ta, 6, input_shape)

    return model_


def get_model_2d_cnn_pho(dataset, input_shape):
    parameters2d_si = {
        'f_batch_size': 4,
        'm_dropout': 0.0,
        'm_filters1': 17,
        'm_filters2': 16,
        'm_hidden_units': 104,
        'm_kernel1_size_x': 2,
        'm_kernel1_size_y': 3,
        'm_kernel2_size_x': 17,
        'm_kernel2_size_y': 10,
        'm_pool1_size_x': 11,
        'm_pool1_size_y': 1,
        'm_pool2_size_x': 16,
        'm_pool2_size_y': 8,
        'm_stride1_size_x': 5,
        'm_stride1_size_y': 3,
        'm_stride2_size_x': 14,
        'm_stride2_size_y': 1
    }

    # parameters2d_ta = {
    #     'f_batch_size': 4,
    #     'm_dropout': 0.2,
    #     'm_filters1': 19,
    #     'm_filters2': 16,
    #     'm_hidden_units': 95,
    #     'm_kernel1_size_x': 2,
    #     'm_kernel1_size_y': 9,
    #     'm_kernel2_size_x': 19,
    #     'm_kernel2_size_y': 19,
    #     'm_pool1_size_x': 4,
    #     'm_pool1_size_y': 1,
    #     'm_pool2_size_x': 18,
    #     'm_pool2_size_y': 1,
    #     'm_stride1_size_x': 1,
    #     'm_stride1_size_y': 4,
    #     'm_stride2_size_x': 3,
    #     'm_stride2_size_y': 2
    # }
    # parameters2d_ta = {'f_batch_size': 1, 'm_dropout': 0.2, 'm_filters1': 16, 'm_filters2': 15,
    #                    'm_hidden_units': 112,
    #                    'm_kernel1_size_x': 2, 'm_kernel1_size_y': 3, 'm_kernel2_size_x': 15,
    #                    'm_kernel2_size_y': 15,
    #                    'm_pool1_size_x': 13, 'm_pool1_size_y': 2, 'm_pool2_size_x': 19, 'm_pool2_size_y': 7,
    #                    'm_stride1_size_x': 2, 'm_stride1_size_y': 4, 'm_stride2_size_x': 8,
    #                    'm_stride2_size_y': 8}
    parameters2d_ta = {'f_batch_size': 1.0, 'm_dropout': 0.05, 'm_filters1': 17.0, 'm_filters2': 16.0,
                       'm_hidden_units': 105.0, 'm_kernel1_size_x': 3.0, 'm_kernel1_size_y': 9.0,
                       'm_kernel2_size_x': 14.0, 'm_kernel2_size_y': 19.0, 'm_pool1_size_x': 20.0,
                       'm_pool1_size_y': 1.0, 'm_pool2_size_x': 10.0, 'm_pool2_size_y': 4.0, 'm_stride1_size_x': 8.0,
                       'm_stride1_size_y': 3.0, 'm_stride2_size_x': 17.0, 'm_stride2_size_y': 7.0}


    print('Building model for ', dataset, ' dataset')
    if dataset == 'si':
        model_ = get_cnn_2d_model3(parameters2d_si, 6, input_shape)
    else:
        model_ = get_cnn_2d_model3(parameters2d_ta, 6, input_shape)

    return model_


def get_model_2d_cnn_wav2vec(dataset, input_shape):
    parameters2d_si = {
        'f_batch_size': 4,
        'm_dropout': 0.0,
        'm_filters1': 17,
        'm_filters2': 16,
        'm_hidden_units': 104,
        'm_kernel1_size_x': 2,
        'm_kernel1_size_y': 3,
        'm_kernel2_size_x': 17,
        'm_kernel2_size_y': 10,
        'm_pool1_size_x': 11,
        'm_pool1_size_y': 1,
        'm_pool2_size_x': 16,
        'm_pool2_size_y': 8,
        'm_stride1_size_x': 5,
        'm_stride1_size_y': 3,
        'm_stride2_size_x': 14,
        'm_stride2_size_y': 1
    }

    # parameters2d_ta = {
    #     'f_batch_size': 4,
    #     'm_dropout': 0.2,
    #     'm_filters1': 19,
    #     'm_filters2': 16,
    #     'm_hidden_units': 95,
    #     'm_kernel1_size_x': 2,
    #     'm_kernel1_size_y': 9,
    #     'm_kernel2_size_x': 19,
    #     'm_kernel2_size_y': 19,
    #     'm_pool1_size_x': 4,
    #     'm_pool1_size_y': 1,
    #     'm_pool2_size_x': 18,
    #     'm_pool2_size_y': 1,
    #     'm_stride1_size_x': 1,
    #     'm_stride1_size_y': 4,
    #     'm_stride2_size_x': 3,
    #     'm_stride2_size_y': 2
    # }

    # wav2vec workign one with 73% acc
    # parameters2d_ta = {'f_batch_size': 2,
    #                    'm_dropout': 0.0,
    #                    'm_filters1': 14,
    #                    'm_filters2': 8,
    #                    'm_hidden_units': 102,
    #                    'm_kernel1_size_x': 9,
    #                    'm_kernel1_size_y': 6,
    #                    'm_kernel2_size_x': 19,
    #                    'm_kernel2_size_y': 18,
    #                    'm_pool1_size_x': 19,
    #                    'm_pool1_size_y': 1,
    #                    'm_pool2_size_x': 13,
    #                    'm_pool2_size_y': 2,
    #                    'm_stride1_size_x': 10,
    #                    'm_stride1_size_y': 3,
    #                    'm_stride2_size_x': 16,
    #                    'm_stride2_size_y': 8}
    parameters2d_ta = {'f_batch_size': 1, 'm_dropout': 0.05, 'm_filters1': 17, 'm_filters2': 16,
                       'm_hidden_units': 105, 'm_kernel1_size_x': 3, 'm_kernel1_size_y': 9,
                       'm_kernel2_size_x': 14, 'm_kernel2_size_y': 19, 'm_pool1_size_x': 20,
                       'm_pool1_size_y': 1, 'm_pool2_size_x': 10, 'm_pool2_size_y': 4, 'm_stride1_size_x': 8,
                       'm_stride1_size_y': 3, 'm_stride2_size_x': 17, 'm_stride2_size_y': 7}

    print('Building model for ', dataset, ' dataset')
    if dataset == 'si':
        model_ = get_cnn_2d_model3(parameters2d_si, 6, input_shape)
    else:
        model_ = get_cnn_2d_model3(parameters2d_ta, 6, input_shape)

    return model_


def get_fnn_model2(param, num_of_classes):
    model_ = Sequential()
    model_.add(Dense(param['dense1'], input_shape=(555, 29), activation='relu'))
    # model.add(Dropout(0.3))
    model_.add(Flatten())
    model_.add(Dropout(param['m_dropout']))
    model_.add(Dense(param['dense2'], activation='relu'))
    model_.add(Dropout(param['m_dropout']))
    model_.add(Dense(param['dense3'], activation='relu'))
    model_.add(Dropout(param['m_dropout']))
    model_.add(Dense(num_of_classes, activation='softmax'))

    # Compile model
    model_.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model_.summary()
    return model_


def get_model_fnn2(dataset):
    parameters2d_si = {
        'dense1': 705,
        'dense2': 354,
        'dense3': 252,
        'f_batch_size': 22,
        'm_dropout': 0.1
    }

    parameters2d_ta = {
        'dense1': 508,
        'dense2': 221,
        'dense3': 205,
        'f_batch_size': 1.0,
        'm_dropout': 0.25
    }

    print('Building model for ', dataset, ' dataset')
    if dataset == 'si':
        model_ = get_fnn_model2(parameters2d_si, 6)
    else:
        model_ = get_fnn_model2(parameters2d_ta, 6)

    return model_


def get_fnn_mfcc_model(param, num_of_classes):
    model_ = Sequential()
    model_.add(
        Dense(param['dense1'], input_shape=(600 * 13,), activation='relu', kernel_regularizer=regularizers.l1(0.001)))
    model_.add(Dropout(param['m_dropout']))
    model_.add(Dense(param['dense2'], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_.add(Dropout(param['m_dropout']))
    model_.add(Dense(param['dense3'], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_.add(Dropout(param['m_dropout']))
    model_.add(Dense(param['dense4'], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_.add(Dropout(param['m_dropout']))
    model_.add(Dense(param['dense5'], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_.add(Dropout(param['m_dropout']))
    model_.add(Dense(num_of_classes, activation='softmax'))

    # Compile model
    model_.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model_.summary()
    return model_


def get_model_fnn_mfcc(dataset):
    parameters2d = {
        'dense1': 1000,
        'dense2': 500,
        'dense3': 500,
        'dense4': 500,
        'dense5': 100,
        'f_batch_size': 128,
        'm_dropout': 0.25
    }

    print('Building model for ', dataset, ' dataset')
    model_ = get_fnn_mfcc_model(parameters2d, 6)

    return model_


def get_svm(class_weights):
    print('creating svm model')
    clf = svm.SVC(
        kernel='rbf',
        C=1000.0,
        gamma='auto',
        decision_function_shape="ovr",
        class_weight=class_weights
    )
    print(clf.get_params())
    return clf


# ----------------------------------------------------------------
model_dic = {
    'ds1': {
        '1d_cnn': get_model_1d_cnn,
        '2d_cnn': get_model_2d_cnn,
        'svm': get_svm
    },
    'ds2': {
        '1d_cnn': get_model_1d_cnn_ds2,
        '2d_cnn': get_model_2d_cnn_ds2,
        'svm': get_svm
    },
    'phonemes': {
        '1d_cnn': get_model_1d_cnn_pho,
        '2d_cnn': get_model_2d_cnn_pho,
        'svm': get_svm
    },
    'wav2vec': {
        '1d_cnn': get_model_1d_cnn_wav2vec,
        '2d_cnn': get_model_2d_cnn_wav2vec,
        'svm': get_svm
    }
}

# ----------------------------------------------------------------
# K-fold trainig process
splits = 5
model_fn = model_dic[expr][mtype]

classes = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5']

X, Y = get_data(lang, expr, mtype)
# X, Y = get_mfcc_data(dataset)

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_seed)

cvscores = []
c_reports = []

fold = 0
for train, test in kfold.split(X, np.argmax(Y, axis=1)):
    fold += 1
    print('Fold ', fold)
    # create model
    if mtype != 'svm':
        model = model_fn(lang, X[train].shape[1:])
        # Fit the model
        model.fit(
            X[train], Y[train],
            validation_data=(X[test], Y[test]),
            epochs=9, batch_size=4, verbose=2
        )

        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        y_pedict = model.predict(X[test])
    else:
        class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(np.argmax(Y[train], axis=1)),
            np.argmax(Y[train], axis=1)
        )
        class_weights = dict(enumerate(class_weights))
        model = get_svm(class_weights)
        model.fit(
            # np.reshape(X[train], (X[train].shape[0], X[train].shape[1]*X[train].shape[2])),
            X[train],
            np.argmax(Y[train], axis=1)
        )
        # evaluate the model
        scores = 0, model.score(
            # np.reshape(X[test], (X[test].shape[0], X[test].shape[1]*X[test].shape[2])),
            X[test],
            np.argmax(Y[test], axis=1)
        )
        y_pedict = model.decision_function(
            # np.reshape(X[test], (X[test].shape[0], X[test].shape[1]*X[test].shape[2]))
            X[test]
        )
    # class_weights = class_weight.compute_class_weight(
    #     'balanced',
    #     np.unique(np.argmax(Y[train], axis=1)),
    #     np.argmax(Y[train], axis=1)
    # )
    # class_weights = dict(enumerate(class_weights))
    # model = get_svm(class_weights)
    # model.fit(
    #     # np.reshape(X[train], (X[train].shape[0], X[train].shape[1]*X[train].shape[2])),
    #     X[train],
    #     np.argmax(Y[train], axis=1)
    # )
    # evaluate the model
    # scores = model.evaluate(X[test], Y[test], verbose=0)
    # scores = 0, model.score(
    #     # np.reshape(X[test], (X[test].shape[0], X[test].shape[1]*X[test].shape[2])),
    #     X[test],
    #     np.argmax(Y[test], axis=1)
    # )
    cvscores.append(scores)
    # y_pedict = model.predict(X[test])
    # y_pedict = model.decision_function(
    #     # np.reshape(X[test], (X[test].shape[0], X[test].shape[1]*X[test].shape[2]))
    #     X[test]
    # )
    print(y_pedict.shape)
    print(np.unique(y_pedict))
    print(np.unique(np.argmax(Y[test], axis=1)))
    report = classification_report(np.argmax(Y[test], axis=1), np.argmax(y_pedict, axis=1), target_names=classes,
                                   output_dict=True)
    df = pd.DataFrame(report).T
    c_reports.append(df)
    # clear tf graphs
    K.clear_session()

print('---Results----')

# for i in range(splits):
#     print('Fold     :', i+1)
#     print('Accuracy :',cvscores[i][1])
#     print('Loss :',cvscores[i][0])
#     print('Classification Report :')
#     print(c_reports[i])
#     print('------------')

print('---Overall Data---')

cvscores = pd.DataFrame(cvscores, columns=['loss', 'acc'])
print('mean')
print(cvscores.mean(axis=0))
print('std')
print(cvscores.std(axis=0))

# print()
# print('classification report')
# o_report =pd.concat(c_reports)
# print('mean')
# print(o_report.groupby(level=0).mean())
# print('std')
# print(o_report.groupby(level=0).std())
# print('----------------------')

# -----------------------------------------------------------

# iterations = 10
# random_samples = 10

# splits= 5
# model_fn = model_dic[expr][mtype]

# scores= np.zeros(
#         (2, iterations, random_samples, splits)
#     )

# classes = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5']

# X, Y = get_data(lang, expr, mtype)

# for i in range(iterations):
#     print('Iteration ', i)
#     sample_ratio = (500*(i+1))/X.shape[0]
#     print('Sample Ratio ', sample_ratio)
#     for s in range(random_samples):
#         print('Sample ', s)
#         x1, _, y1, _ = train_test_split(
#                 X, Y,
#                 train_size=sample_ratio,
#                 random_state=s,
#                 shuffle=True
#             )
#         print('Sample size ', x1.shape)
#         # define 10-fold cross validation test harness
#         kfold = StratifiedKFold(n_splits=splits, random_state=random_seed)

#         fold = 0
#         for train, test in kfold.split(x1, np.argmax(y1, axis=1)):
#             fold += 1
#             print('Fold ', fold)

#             earlyStopping = EarlyStopping(
#                 monitor='val_loss', patience=1, verbose=1, mode='min',
#                 restore_best_weights=True
#             )

#             # create model
#             model = model_fn(lang, X[train].shape[1:])
#             # Fit the model
#             model.fit(
#                 x1[train], y1[train],
#                 validation_data=(x1[test], y1[test]),
#                 epochs=3, batch_size=4, verbose=2,
#                 callbacks=[earlyStopping]
#             )
#             # class_weights = class_weight.compute_class_weight(
#             #     'balanced',
#             #     np.unique(np.argmax(Y[train], axis=1)),
#             #     np.argmax(Y[train], axis=1)
#             # )
#             # class_weights = dict(enumerate(class_weights))
#             # model = get_svm(class_weights)
#             # model.fit(
#             #     # np.reshape(X[train], (X[train].shape[0], X[train].shape[1]*X[train].shape[2])),
#             #     X[train],
#             #     np.argmax(Y[train], axis=1)
#             # )
#             # evaluate the model
#             score = model.evaluate(x1[test], y1[test], verbose=0)
#             # scores = 0, model.score(
#             #     # np.reshape(X[test], (X[test].shape[0], X[test].shape[1]*X[test].shape[2])),
#             #     X[test],
#             #     np.argmax(Y[test], axis=1)
#             # )
#             print('Score :',score)
#             scores[0, i, s, fold-1] = score[0]
#             scores[1, i, s, fold-1] = score[1]
#             # cvscores.append(scores)
#             # y_pedict = model.predict(X[test])
#             # y_pedict = model.decision_function(
#             #     # np.reshape(X[test], (X[test].shape[0], X[test].shape[1]*X[test].shape[2]))
#             #     X[test]
#             # )
#             # print(y_pedict.shape)
#             # print(np.unique(y_pedict))
#             # print(np.unique(np.argmax(Y[test], axis=1)))
#             # report = classification_report(np.argmax(Y[test], axis=1), np.argmax(y_pedict, axis=1), target_names=classes, output_dict=True)
#             # df = pd.DataFrame(report).T
#             # c_reports.append(df)
#             # clear tf graphs
#             K.clear_session()

# #print results
# # print(scores)
# for i in range(iterations):
#     print('Iteration:', i)
#     print('mean :',  ' loss:', np.mean(scores[0, i]), ' acc:', np.mean(scores[1, i]))
#     print('std :',  ' loss:', np.std(scores[0, i]), ' acc:', np.std(scores[1, i]))
