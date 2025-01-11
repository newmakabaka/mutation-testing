# cifar-10数据集：有60000张图片，其中50000张训练图片 和 10000张测试图片
# 图片尺寸为32 * 32 * 3（彩色图片）
# 包含10个类别：飞机、汽车、鸟、猫、鹿、狗、蛙、马、船、卡车

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
    for op in range(0,5):
        for k in range(10):
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train=x_train.reshape(-1,32,32,3)
            x_test=x_test.reshape(-1,32,32,3)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            y_train = y_train.reshape(y_train.shape[0])
            y_test = y_test.reshape(y_test.shape[0])

            cate_index = np.where(y_train == k)
            mutate_images, mutate_labels = source_level_operators(x_train, y_train, op, cate_index)
            # random_index = random.sample(range(x_train.shape[0]), cate_index[0].shape[0])
            # mutate_images, mutate_labels = source_level_operators(x_train, y_train, op, random_index)  # 随机选择的输入

            mutate_labels = np_utils.to_categorical(mutate_labels, 10)
            y_test = np_utils.to_categorical(y_test, 10)

            model=Sequential([
                Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)),
                Conv2D(64, (3, 3), activation='relu', padding="same"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(128, (3, 3), activation='relu', padding="same"),
                Conv2D(128, (3, 3), activation='relu', padding="same"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(256,activation='relu'),
                Dropout(0.5),
                Dense(256, activation='relu'),
                Dense(10,activation='softmax')
            ])

            # print(model.summary())
            print("mutant {} category {} start train".format(operator_name[op], k))
            model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.05),metrics=['accuracy'])
            model.fit(
                mutate_images,
                mutate_labels,
                batch_size=200,
                epochs=80,
                shuffle=True,
                verbose=1,
                validation_data=(x_test,y_test)
            )
            # model.save("./model/model_cifar10_unknow.h5")
            model.save("../mutants/cifar10_unknow/source_level_mutants/mutant_{}_category{}.h5".format(operator_name[op], k))    # 对应类
            # model.save("../random_mutants/cifar10_unknow/source_level_mutants/mutant_{}_{}.h5".format(operator_name[op], k))  # 随机
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

    model=load_model("./model/Adapt_cifar10_unknow.h5")
    model.summary()
    test_score=model.evaluate(x_test,y_test,verbose=1)
    print('test score:',test_score)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", required=True, type=str)
    # args = parser.parse_args()
    # train()
    test()
