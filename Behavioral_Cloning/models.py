from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Lambda, Cropping2D
from keras.layers import MaxPooling2D, ELU, Dropout
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
import gc

# Function to define and fit LeNet-5 based model
def train_model_lenet(train_generator, val_generator, num_train, num_val):
    # Model definition
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50,28), (0,0))))
    model.add(Convolution2D(6,5,5, activation="elu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation="elu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(10))
    model.add(Dense(1))

    # Save visualization of network architecture
    plot(model, to_file='images/lenet_architecture.png', show_shapes=True)
    print("Network architecture saved at: images/lenet_architecture.png")

    # Fit model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, \
                                        samples_per_epoch=num_train, \
                                        validation_data=val_generator, \
                                        nb_val_samples=num_val, \
                                        nb_epoch=5, verbose=1)

    # Save model
    model.save('models/model_lenet.h5')
    print("Model saved at: models/model_lenet.h5")
    gc.collect()

# Function to define and fit Nvidia model
def train_model_nvidia(train_generator, val_generator, num_train, num_val):
    # Model definition
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0 - 0.5), input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50,28), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # Save visualization of network architecture
    plot(model, to_file='images/nvidia_architecture.png', show_shapes=True)
    print("Network architecture saved at: images/nvidia_architecture.png")

    # Fit model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, \
                                        samples_per_epoch=num_train, \
                                        validation_data=val_generator, \
                                        nb_val_samples=num_val, \
                                        nb_epoch=5, verbose=1)

    # Save model
    model.save('models/model_nvidia.h5')
    print("Model saved at: models/model_nvidia.h5")
    gc.collect()

'''
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model Mean Squared Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
'''

# Function to define and fit LeNet-5 based model
def train_model_comma(train_generator, val_generator, num_train, num_val):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50,28), (0,0))))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    # Save visualization of network architecture
    plot(model, to_file='images/comma_architecture.png', show_shapes=True)
    print("Network architecture saved at: images/comma_architecture.png")

    # Fit model
    model.compile(loss="mse", optimizer="adam")
    history_object = model.fit_generator(train_generator, \
                                        samples_per_epoch=num_train, \
                                        validation_data=val_generator, \
                                        nb_val_samples=num_val, \
                                        nb_epoch=5, verbose=1)

    # Save model
    model.save('models/model_comma.h5')
    print("Model saved at: models/model_comma.h5")
