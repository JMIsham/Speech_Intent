# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input/speechta"))

# Any results you write to the current directory are saved as output.

# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from datetime import datetime

from keras import backend as K
from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.pyll.base import scope

# --------------------------------------------------------------------------

lang = 'ta'  # 'si', 'ta'
expr = 'wav2vec'  # 'ds1', 'ds2', 'phonemes', wav2vec
mtype = '2d_cnn'  # '1d_cnn', '2d_cnn
merger = 'ds2'  # 'ds1', 'ds2', 'phonemes', wav2vec

csv_dic = {
    'si': '../input/speechsi/formatted_data_v2.csv',
    'ta': '../data/Tamil_Dataset/Tamil_Data.csv'
}

datax_dic = {
    'si': {
        'ds1': '../input/speechsi/ds_decode_data_v2_padded.npy',
        'ds2': '../input/speechds2/ds2_decode_sinhala_data_v2_padded.npy',
        'ds2': '../input/speechds2/ds2_decode_sinhala_data_v2_padded.npy',
        'phonemes': '../input/speech-phonemes/phoneme_decode_sinhala_data_v2_padded.npy'
    },
    'ta': {
        'ds1': '../input/speechta/ds_decode_tamil_data_v2_padded.npy',
        'ds2': '../data/Tamil_Dataset/ds2_decode_Tamil_data_v2.npy',
        'wav2vec': '../data/Tamil_Dataset/wav2vec_decode_Tamil_data_v2.npy',
        'phonemes': '../data/Tamil_Dataset/phoneme_decode_Tamil_data_v2.npy'
    }
}

# random number
random_seed = 7
np.random.seed(random_seed)

# read csv and gey classes
data = pd.read_csv(csv_dic[lang])

data_y = data['intent']
indices = np.arange(len(data_y))

# load ds decoded output
data_x = np.load(datax_dic[lang][expr], allow_pickle=True)
data_x_merge = np.load(datax_dic[lang][merger], allow_pickle=True)

# data_merge  = np.array()
# for i in range(0, data_x.shape[0]):
#     data_merge.append(np.concatenate(data_x[i], data_x_merge[i], axis=1))



# print("*********** : ", data_x_merge.shape, data_x.shape, np.concatenate((data_x, data_x_merge), axis=1).shape)
# data_x = np.concatenate((data_x, data_x_merge), axis=1)

num_of_classes = len(data_y.unique())
print('Total data samples   :', data_x.shape[0])
print('Number of classes    :', num_of_classes)

# one hot encoding y values
data_y = pd.get_dummies(data_y)
# data_y['id'] = indices
data_y = data_y.values

# calculate total occurrences of classes
print('Occurrences of classes in total dataset')
print(data_y[:, 0:6].sum(axis=0))

# split data
if lang == 'si':
    data_x_train, data_x_test, data_y_train, data_y_test = \
        train_test_split(data_x, data_y, test_size=0.2, random_state=random_seed)
    data_x_dev, data_x_test, data_y_dev, data_y_test = \
        train_test_split(data_x_test, data_y_test, test_size=0.2, random_state=random_seed)
else:
    data_x_train, data_x_test, data_x_merge_train, data_x_merge_test, data_y_train, data_y_test = \
        train_test_split(data_x, data_x_merge, data_y, test_size=0.2, random_state=random_seed)

# reshape data for CNN
if lang == 'si':
    if mtype == '1d_cnn':
        data_x_train = np.reshape(data_x_train, (data_x_train.shape[0], data_x_train.shape[1], data_x_train.shape[2]))
        data_x_dev = np.reshape(data_x_dev, (data_x_dev.shape[0], data_x_dev.shape[1], data_x_dev.shape[2]))
        data_x_test = np.reshape(data_x_test, (data_x_test.shape[0], data_x_test.shape[1], data_x_test.shape[2]))
    if mtype == '2d_cnn':
        data_x_train = np.reshape(data_x_train,
                                  (data_x_train.shape[0], data_x_train.shape[1], data_x_train.shape[2], 1))
        data_x_dev = np.reshape(data_x_dev, (data_x_dev.shape[0], data_x_dev.shape[1], data_x_dev.shape[2], 1))
        data_x_test = np.reshape(data_x_test, (data_x_test.shape[0], data_x_test.shape[1], data_x_test.shape[2], 1))
else:
    if mtype == '1d_cnn':
        data_x_train = np.reshape(data_x_train, (data_x_train.shape[0], data_x_train.shape[1], data_x_train.shape[2]))
        data_x_test = np.reshape(data_x_test, (data_x_test.shape[0], data_x_test.shape[1], data_x_test.shape[2]))
    if mtype == '2d_cnn':
        data_x_train = np.reshape(data_x_train,
                                  (data_x_train.shape[0], data_x_train.shape[1], data_x_train.shape[2], 1))
        data_x_test = np.reshape(data_x_test, (data_x_test.shape[0], data_x_test.shape[1], data_x_test.shape[2], 1))
        data_x_merge_train = np.reshape(data_x_merge_train,
                                  (data_x_merge_train.shape[0], data_x_merge_train.shape[1], data_x_merge_train.shape[2], 1))
        data_x_merge_test = np.reshape(data_x_merge_test, (data_x_merge_test.shape[0], data_x_merge_test.shape[1], data_x_merge_test.shape[2], 1))

# # Separate indices
# indices_train = data_y_train[:, 6]
# data_y_train = data_y_train[:, 0:6]
# # indices_dev = data_y_dev[:, 6]
# # data_y_dev = data_y_dev[:, 0:6]
# indices_test = data_y_test[:, 6]
# data_y_test = data_y_test[:, 0:6]

# print data set sizes
print()
print('----Dataset Detail----')
print('Training Data X  :', data_x_train.shape, ' Y', data_y_train.shape)
if lang == 'si':
    print('Validation Data X:', data_x_dev.shape, ' Y', data_y_dev.shape)
print('Testing Data X   :', data_x_test.shape, ' Y', data_y_test.shape)
print()

print('--Occurrences in each classes--')
print('Training Data   :', data_y_train.sum(axis=0))
if lang == 'si':
    print('Validation Data :', data_y_dev.sum(axis=0))
print('Testing Data    :', data_y_test.sum(axis=0))


# ---------------------------------------------------------------------
def get_cnn1d_model3(param, num_of_classes):
    model_ = Sequential()

    model_.add(Conv1D(
        param['m_filters1'],
        param['m_kernel1_size'],
        activation='relu',
        padding='same',
        # input_shape=(256, 42)
        input_shape=data_x_train.shape[1:]
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


def get_cnn1d_search_space3():
    space = {
        # "lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-3)), # 0.0001 - 0.001
        "m_filters1": scope.int(hp.quniform("m_filters1", 5, 50, 1)),
        # "m_kernel1_size_x": scope.int(hp.quniform("m_kernel1_size_x", 1, 10, 1)),
        # "m_kernel1_size_y": scope.int(hp.quniform("m_kernel1_size_y", 1, 10, 1)),
        "m_kernel1_size": scope.int(hp.quniform("m_kernel1_size", 2, 50, 1)),
        "m_filters2": scope.int(hp.quniform("m_filters2", 5, 50, 1)),
        # "m_kernel2_size_x": scope.int(hp.quniform("m_kernel2_size_x", 5, 20, 1)),
        # "m_kernel2_size_y": scope.int(hp.quniform("m_kernel2_size_y", 5, 20, 1)),
        "m_kernel2_size": scope.int(hp.quniform("m_kernel2_size", 2, 50, 1)),
        # "m_pool1_size_x": scope.int(hp.quniform("m_pool1_size_x", 3, 20, 1)),
        # "m_pool1_size_y": scope.int(hp.quniform("m_pool1_size_y", 1, 10, 1)),
        "m_pool1_size": scope.int(hp.quniform("m_pool1_size", 2, 50, 1)),
        # "m_pool2_size_x": scope.int(hp.quniform("m_pool2_size_x", 3, 20, 1)),
        # "m_pool2_size_y": scope.int(hp.quniform("m_pool2_size_y", 1, 10, 1)),
        "m_pool2_size": scope.int(hp.quniform("m_pool2_size", 2, 50, 1)),
        # "m_stride1_size_x": scope.int(hp.quniform("m_stride1_size_x", 1, 20, 1)),
        # "m_stride1_size_y": scope.int(hp.quniform("m_stride1_size_y", 1, 10, 1)),
        "m_stride1_size": scope.int(hp.quniform("m_stride1_size", 1, 40, 1)),
        # "m_stride2_size_x": scope.int(hp.quniform("m_stride2_size_x", 1, 20, 1)),
        # "m_stride2_size_y": scope.int(hp.quniform("m_stride2_size_y", 1, 10, 1)),
        "m_stride2_size": scope.int(hp.quniform("m_stride2_size", 1, 40, 1)),
        "m_hidden_units": scope.int(hp.quniform("m_hidden_units", 40, 200, 1)),
        "m_dropout": hp.quniform("m_dropout", 0, 0.5, 0.05),

        "f_epochs": 2,
        "f_batch_size": scope.int(hp.quniform("f_batch_size", 1, 10, 1))
    }

    return space


def get_cnn1_model3(param, num_of_classes):
    model_ = Sequential()

    model_.add(Conv2D(
        param['m_filters1'],
        (param['m_kernel1_size_x'], param['m_kernel1_size_y']),
        activation='relu',
        padding='same',
        # input_shape=(555, 29, 1)
        # input_shape=(256, 42, 1)
        input_shape=data_x_train.shape[1:]
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


def get_cnn1_model3_merge(param, num_of_classes):

    visible1 = Input(shape=data_x_train.shape[1:])
    conv11 = Conv2D(
        param['m_filters1'],
        (param['m_kernel1_size_x'], param['m_kernel1_size_y']),
        activation='relu',
        padding='same',
        # input_shape=(555, 29, 1)
        # input_shape=(256, 42, 1)
    )(visible1)
    pool11 = MaxPooling2D(
        pool_size=(param['m_pool1_size_x'], param['m_pool1_size_y']),
        strides=(param['m_stride1_size_x'], param['m_stride1_size_y'])
    )(conv11)
    drop11 = Dropout(
        param['m_dropout']
    )(pool11)

    conv21 = Conv2D(
        param['m_filters2'],
        (param['m_kernel2_size_x'], param['m_kernel2_size_y']),
        activation='relu',
        padding='same'
    )(drop11)

    pool21 = MaxPooling2D(
        pool_size=(param['m_pool2_size_x'], param['m_pool2_size_y']),
        strides=(param['m_stride2_size_x'], param['m_stride2_size_y'])
    )(conv21)

    drop21 = Dropout(
        param['m_dropout']
    )(pool21)

    flatten1 = Flatten()(drop21)

    visible2 = Input(shape=data_x_merge_train.shape[1:])
    conv12 = Conv2D(
        param['m_filters1_2'],
        (param['m_kernel1_size_x_2'], param['m_kernel1_size_y_2']),
        activation='relu',
        padding='same',
        # input_shape=(555, 29, 1)
        # input_shape=(256, 42, 1)
    )(visible2)
    pool12 = MaxPooling2D(
        pool_size=(param['m_pool1_size_x_2'], param['m_pool1_size_y_2']),
        strides=(param['m_stride1_size_x_2'], param['m_stride1_size_y_2'])
    )(conv12)
    drop12 = Dropout(
        param['m_dropout_2']
    )(pool12)

    conv22 = Conv2D(
        param['m_filters2_2'],
        (param['m_kernel2_size_x_2'], param['m_kernel2_size_y_2']),
        activation='relu',
        padding='same'
    )(drop12)

    pool22 = MaxPooling2D(
        pool_size=(param['m_pool2_size_x_2'], param['m_pool2_size_y_2']),
        strides=(param['m_stride2_size_x_2'], param['m_stride2_size_y_2'])
    )(conv22)

    drop22 = Dropout(
        param['m_dropout_2']
    )(pool22)

    flatten2 = Flatten()(drop22)

    merge = concatenate([flatten1, flatten2])

    hidden1 = Dense(
        param['m_hidden_units'],
        activation='relu'
    )(merge)

    output = Dense(
        num_of_classes,
        activation='softmax'
    )(hidden1)

    model = Model(inputs=[visible1, visible2], outputs=output)

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

    return model


def get_cnn1_search_space3():
    space = {
        # "lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-3)), # 0.0001 - 0.001
        "m_filters1": scope.int(hp.quniform("m_filters1", 10, 20, 1)),
        "m_kernel1_size_x": scope.int(hp.quniform("m_kernel1_size_x", 1, 10, 1)),
        "m_kernel1_size_y": scope.int(hp.quniform("m_kernel1_size_y", 1, 10, 1)),
        "m_filters2": scope.int(hp.quniform("m_filters2", 5, 18, 1)),
        "m_kernel2_size_x": scope.int(hp.quniform("m_kernel2_size_x", 5, 20, 1)),
        "m_kernel2_size_y": scope.int(hp.quniform("m_kernel2_size_y", 5, 20, 1)),
        "m_pool1_size_x": scope.int(hp.quniform("m_pool1_size_x", 3, 20, 1)),
        "m_pool1_size_y": scope.int(hp.quniform("m_pool1_size_y", 1, 10, 1)),
        "m_pool2_size_x": scope.int(hp.quniform("m_pool2_size_x", 3, 20, 1)),
        "m_pool2_size_y": scope.int(hp.quniform("m_pool2_size_y", 1, 10, 1)),
        "m_stride1_size_x": scope.int(hp.quniform("m_stride1_size_x", 1, 20, 1)),
        "m_stride1_size_y": scope.int(hp.quniform("m_stride1_size_y", 1, 10, 1)),
        "m_stride2_size_x": scope.int(hp.quniform("m_stride2_size_x", 1, 20, 1)),
        "m_stride2_size_y": scope.int(hp.quniform("m_stride2_size_y", 1, 10, 1)),
        "m_hidden_units": scope.int(hp.quniform("m_hidden_units", 95, 130, 1)),
        "m_dropout": hp.quniform("m_dropout", 0, 0.5, 0.05),

        "f_epochs": 2,
        "f_batch_size": scope.int(hp.quniform("f_batch_size", 0, 10, 1))
    }

    return space


def get_cnn1_search_space3_merge():
    space = {
        # "lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-3)), # 0.0001 - 0.001
        "m_filters1": scope.int(hp.quniform("m_filters1", 10, 20, 1)),
        "m_kernel1_size_x": scope.int(hp.quniform("m_kernel1_size_x", 1, 10, 1)),
        "m_kernel1_size_y": scope.int(hp.quniform("m_kernel1_size_y", 1, 10, 1)),
        "m_filters2": scope.int(hp.quniform("m_filters2", 5, 18, 1)),
        "m_kernel2_size_x": scope.int(hp.quniform("m_kernel2_size_x", 5, 20, 1)),
        "m_kernel2_size_y": scope.int(hp.quniform("m_kernel2_size_y", 5, 20, 1)),
        "m_pool1_size_x": scope.int(hp.quniform("m_pool1_size_x", 3, 20, 1)),
        "m_pool1_size_y": scope.int(hp.quniform("m_pool1_size_y", 1, 10, 1)),
        "m_pool2_size_x": scope.int(hp.quniform("m_pool2_size_x", 3, 20, 1)),
        "m_pool2_size_y": scope.int(hp.quniform("m_pool2_size_y", 1, 10, 1)),
        "m_stride1_size_x": scope.int(hp.quniform("m_stride1_size_x", 1, 20, 1)),
        "m_stride1_size_y": scope.int(hp.quniform("m_stride1_size_y", 1, 10, 1)),
        "m_stride2_size_x": scope.int(hp.quniform("m_stride2_size_x", 1, 20, 1)),
        "m_stride2_size_y": scope.int(hp.quniform("m_stride2_size_y", 1, 10, 1)),

        "m_filters1_2": scope.int(hp.quniform("m_filters1_2", 10, 20, 1)),
        "m_kernel1_size_x_2": scope.int(hp.quniform("m_kernel1_size_x_2", 1, 10, 1)),
        "m_kernel1_size_y_2": scope.int(hp.quniform("m_kernel1_size_y_2", 1, 10, 1)),
        "m_filters2_2": scope.int(hp.quniform("m_filters2_2", 5, 18, 1)),
        "m_kernel2_size_x_2": scope.int(hp.quniform("m_kernel2_size_x_2", 5, 20, 1)),
        "m_kernel2_size_y_2": scope.int(hp.quniform("m_kernel2_size_y_2", 5, 20, 1)),
        "m_pool1_size_x_2": scope.int(hp.quniform("m_pool1_size_x_2", 3, 20, 1)),
        "m_pool1_size_y_2": scope.int(hp.quniform("m_pool1_size_y_2", 1, 10, 1)),
        "m_pool2_size_x_2": scope.int(hp.quniform("m_pool2_size_x_2", 3, 20, 1)),
        "m_pool2_size_y_2": scope.int(hp.quniform("m_pool2_size_y_2", 1, 10, 1)),
        "m_stride1_size_x_2": scope.int(hp.quniform("m_stride1_size_x_2", 1, 20, 1)),
        "m_stride1_size_y_2": scope.int(hp.quniform("m_stride1_size_y_2", 1, 10, 1)),
        "m_stride2_size_x_2": scope.int(hp.quniform("m_stride2_size_x_2", 1, 20, 1)),
        "m_stride2_size_y_2": scope.int(hp.quniform("m_stride2_size_y_2", 1, 10, 1)),

        # "m_filters2_3": scope.int(hp.quniform("m_filters2_3", 5, 18, 1)),
        # "m_kernel2_size_x_3": scope.int(hp.quniform("m_kernel2_size_x_3", 5, 20, 1)),
        # "m_kernel2_size_y_3": scope.int(hp.quniform("m_kernel2_size_y_3", 5, 20, 1)),
        # "m_pool1_size_x_3": scope.int(hp.quniform("m_pool1_size_x_3", 3, 20, 1)),
        # "m_pool1_size_y_3": scope.int(hp.quniform("m_pool1_size_y_3", 1, 10, 1)),
        # "m_pool2_size_x_3": scope.int(hp.quniform("m_pool2_size_x_3", 3, 20, 1)),
        # "m_pool2_size_y_3": scope.int(hp.quniform("m_pool2_size_y_3", 1, 10, 1)),

        "m_hidden_units": scope.int(hp.quniform("m_hidden_units", 95, 130, 1)),
        "m_dropout": hp.quniform("m_dropout", 0, 0.5, 0.05),
        "m_dropout_2": hp.quniform("m_dropout_2", 0, 0.5, 0.05),
        # "m_dropout_3": hp.quniform("m_dropout_3", 0, 0.5, 0.05),

        "f_epochs": 2,
        "f_batch_size": scope.int(hp.quniform("f_batch_size", 0, 10, 1))
    }

    return space


def get_fnn_model1(param, num_of_classes):
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


def get_fnn_model1_search_space():
    space = {
        "dense1": scope.int(hp.quniform("dense1", 400, 800, 1)),
        "dense2": scope.int(hp.quniform("dense2", 200, 500, 1)),
        "dense3": scope.int(hp.quniform("dense3", 10, 300, 1)),
        "m_dropout": hp.quniform("m_dropout", 0, 0.5, 0.05),

        "f_epochs": 2,
        "f_batch_size": scope.int(hp.quniform("f_batch_size", 1, 30, 1))
    }

    return space


# ---------------------------------------------------------------------------------
model_dic = {
    '1d_cnn': {
        'mf': get_cnn1d_model3,
        'sf': get_cnn1d_search_space3
    },
    '2d_cnn': {
        'mf': get_cnn1_model3_merge,
        'sf': get_cnn1_search_space3_merge
    }
}

# ---------------------------------------------------------------------------------

# set model function and hyper. space function

model_f = model_dic[mtype]['mf']
space_f = model_dic[mtype]['sf']


# define objective function
def objective(parameters):
    try:
        model = model_f(
            parameters,
            num_of_classes
        )

        model.fit(
            [data_x_train, data_x_merge_train], data_y_train,
            # [data_x_train, data_x_merge_train], data_y_train,
            # validatmtypeion_data=(data_x_dev, data_y_dev),
            epochs=int(parameters['f_epochs']),
            batch_size=int(parameters['f_batch_size']),
            verbose=0
        )

        # score = model.evaluate(data_x_dev, data_y_dev, batch_size=int(parameters['f_batch_size']), verbose=0)
        score = model.evaluate([data_x_test, data_x_merge_test], data_y_test, batch_size=int(parameters['f_batch_size']), verbose=0)

        print("*********** best score")
        print({
            'loss': score[0],
            'acc': score[1],
            'status': STATUS_OK
        })
        K.clear_session()
        del model

        return {
            'loss': score[0],
            'acc': score[1],
            'status': STATUS_OK
        }

    except Exception as e:
        print(e)
        K.clear_session()
        print(e)

        return {
            'status': STATUS_FAIL
        }


print('Starting Hyperopt')

max_evals = 100

# trials = Trials()
best = fmin(
    objective,
    space=space_f(),
    algo=tpe.suggest,
    max_evals=max_evals,
    verbose=1
)

print('Best parameters')
print(best)

print('Saving Parameters to parameter_file.txt')
string = 'Timestamp   : ' + str(datetime.now()) + ', ' + \
         'Evaluations : ' + str(max_evals) + ', ' + \
         'Best Param  : ' + str(best)
file = open('parameter_file.txt', 'a')
file.write(string)
file.close()