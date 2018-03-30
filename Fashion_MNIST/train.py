import os
import gzip
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
from keras.utils import np_utils
from sklearn.utils import shuffle

# hyperparameters
n_classes = 10
epochs = 30
batch_sz = 128
img_h = 28
img_w = 28


def load_data(path, type='train'):
    # generate file path
    if type =='test':
        type = 't10k'
    label_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % type)
    img_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % type)
    # load data
    with gzip.open(label_path) as lbl_bt:
        y = np.frombuffer(lbl_bt.read(), dtype=np.uint8, offset=8)
        y = y.reshape(-1, 1)
    with gzip.open(img_path) as img_bt:
        x = np.frombuffer(img_bt.read(), dtype=np.uint8, offset=16)
        x = x.reshape(-1, 28, 28, 1)
    return x, y


def LeNet5():
    # define model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='elu',
                            input_shape=(img_h, img_w, 1), init='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu',
                            init='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='elu',
                            init='he_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(300, activation='elu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(150, activation='elu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax', init='he_normal'))

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
    return model


def train():
    # load and normalize data
    x, y = load_data(path='../Datasets/fashion-mnist/data/fashion/', type='train')
    x = x.astype('float32')
    x_norm = x / 255
    #x_std = (x - np.mean(x)) / np.std(x)

    # convert label to one-hot encoded vectors
    y_one_hot = np_utils.to_categorical(y, n_classes)

    #print('x_shape = ', x_std.shape)
    #print('y_shape = ', y_one_hot.shape)

    # load LeNet5 model
    model = LeNet5()

    # fit model
    result = model.fit(x_norm, y_one_hot, validation_split=0.3, nb_epoch=epochs,
                        batch_size = batch_sz, verbose=1)

if __name__=='__main__':
    train()
