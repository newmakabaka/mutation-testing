import argparse
import tensorflow as tf
from keras.datasets import mnist, cifar10, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from high_mutant.source_level_operators import *


def train():
    for op in range(0,5):
        for k in range(10):
            # weight_decay = 0.0005
            epoch_num = 60
            lr = 0.1
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train=x_train.reshape(-1,32,32,3)
            x_test=x_test.reshape(-1,32,32,3)
            #
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            y_train = y_train.reshape(y_train.shape[0])
            y_test = y_test.reshape(y_test.shape[0])

            cate_index = np.where(y_train == k)
            # mutate_images, mutate_labels = source_level_operators(x_train, y_train, op, cate_index)
            random_index=random.sample(range(x_train.shape[0]),cate_index[0].shape[0])
            mutate_images, mutate_labels = source_level_operators(x_train, y_train, op, random_index)     # 随机选择的输入

            mutate_labels = np_utils.to_categorical(mutate_labels, 10)
            y_test = np_utils.to_categorical(y_test, 10)

            model=Sequential()
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2),strides=(2,2)))

            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2),strides=(2,2)))

            model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2),strides=(2,2)))

            model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2),strides=(2,2)))

            model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

            model.add(Flatten())  # 2*2*512
            model.add(Dense(4096, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(4096, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))

            def scheduler(epoch):
                if epoch < epoch_num * 0.4:
                    return lr
                if epoch < epoch_num * 0.8:
                    return lr * 0.1
                return lr * 0.01


            change_lr = LearningRateScheduler(scheduler)

            # print(model.summary())
            print("mutant {} category {} start train".format(operator_name[op], k))
            model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adadelta(learning_rate=lr),metrics=['accuracy'])
            model.fit(
                mutate_images,
                mutate_labels,
                batch_size=200,
                epochs=epoch_num,
                callbacks=[change_lr],
                shuffle=True,
                verbose=1,
                validation_data=(x_test,y_test)
            )

            # model.save("./model/model_cifar10_vgg16temp.h5")
            # model.save("../mutants/cifar10_vgg16/source_level_mutants/mutant_{}_category{}.h5".format(operator_name[op], k))
            model.save("D:/random_mutants/cifar10_vgg16/source_level_mutants/mutant_{}_{}.h5".format(operator_name[op], k))
            print("successfully save the model mutant {} category {}".format(operator_name[op], k))


def test():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # print(x_train.shape)
    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    # print(x_train.shape)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model=load_model("./model/model_cifar10_vgg16.h5")
    print(model.summary())
    test_score=model.evaluate(x_test,y_test,verbose=1)
    print('test score:',test_score)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", required=True, type=str)
    # args = parser.parse_args()
    train()
    # test()
