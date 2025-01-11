import argparse
import tensorflow as tf
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D,BatchNormalization
from keras.regularizers import l2
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from high_mutant.source_level_operators import *


def train():
    for op in range(0,1):
        for k in range(10):
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)

            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")
            x_train /= 255.0
            x_test /= 255.0

            cate_index = np.where(y_train == k)
            # mutate_images, mutate_labels = source_level_operators(x_train.copy(), y_train.copy(), op, cate_index)
            random_index = random.sample(range(x_train.shape[0]), cate_index[0].shape[0])
            mutate_images, mutate_labels = source_level_operators(x_train.copy(), y_train.copy(), op, random_index)  # 随机选择的输入

            mutate_labels = np_utils.to_categorical(mutate_labels, 10)
            # y_train = np_utils.to_categorical(y_train, 10)
            y_test = np_utils.to_categorical(y_test, 10)

            model=Sequential([
                Conv2D(64, (3,3),activation='relu',padding="same", input_shape=(28, 28, 1)),
                Conv2D(64, (3,3), activation='relu', padding="same"),
                MaxPooling2D(pool_size=(2,2)),
                Conv2D(128, (3,3), activation='relu', padding="same"),
                Conv2D(128, (3,3), activation='relu', padding="same"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(320, activation='relu'),
                Dropout(0.5),
                Dense(192,activation='relu'),
                Dense(10,activation='softmax')
            ])

            print(model.summary())
            # print("mutant {} category {} start train".format(operator_name[op], k))
            model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.01),metrics=['accuracy'])
            model.fit(
                mutate_images,
                mutate_labels,
                batch_size=128,
                epochs=70,
                shuffle=True,
                verbose=1,
                validation_data=(x_test, y_test)
            )
            # model.save("./model/model_fashion_mnist_unknow.h5")
            # model.save("../mutants/fashion_mnist_unknow/source_level_mutants/mutant_{}_category{}.h5".format(operator_name[op], k))
            model.save("../random_mutants/fashion_mnist_unknow/source_level_mutants/mutant_{}_{}.h5".format(operator_name[op], k))
            print("successfully save the model mutant {} category {}".format(operator_name[op], k))


def test():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # print(x_train.shape)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    # print(x_train.shape)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model=load_model("./model/DLfuzz_fashion_mnist_unknow - 副本.h5")
    model.summary()
    test_score=model.evaluate(x_test,y_test,verbose=1)
    print('test score:',test_score)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", required=True, type=str)
    # args = parser.parse_args()
    # train()
    test()
