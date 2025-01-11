import argparse
import tensorflow as tf
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D,BatchNormalization
from keras.regularizers import l2
from keras.utils import np_utils
from tensorflow.keras.models import load_model


def train():
    weight_decay = 0.0005
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train=x_train.reshape(-1,32,32,3)
    x_test=x_test.reshape(-1,32,32,3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # model=Sequential()
    # for layer in layers:
    #     model.add(layer)


    model=Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3),
                     kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))

    model.add(Flatten())  # 2*2*512
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))


    print(model.summary())
    model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1),metrics=['accuracy'])
    model.fit(
        x_train,
        y_train,
        batch_size=200,
        epochs=50,
        shuffle=True,
        verbose=1,
        validation_data=(x_test,y_test)
    )
    model.save("./model/model_cifar10_unknow.h5")
    print("successfully save the model")





if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", required=True, type=str)
    # args = parser.parse_args()
    train()
    # test()
