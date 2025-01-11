import numpy as np
import tensorflow as tf
from keras.datasets import mnist, cifar10,fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D,BatchNormalization
from keras.regularizers import l2
from keras.utils import np_utils
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random


def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    test_images = test_images.astype('float32') / 255.0
    train_labels=np_utils.to_categorical(train_labels,10)
    test_labels=np_utils.to_categorical(test_labels,10)
    return train_images,train_labels,test_images,test_labels


def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.reshape((train_images.shape[0],32,32,3))
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.reshape((test_images.shape[0], 32, 32, 3))
    test_images = test_images.astype('float32') / 255.0
    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)
    return train_images, train_labels, test_images, test_labels


def load_fashion_mnist():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    test_images = test_images.astype('float32') / 255.0
    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)
    return train_images,train_labels,test_images,test_labels


model_name = "unknow"
dataset_name = "fashion_mnist"     # mnist cifar10
# generate_method="DLfuzz"    # xplore DLfuzz Adapt

root_model_path = "model/model_"
# root_model_save_path = "model/{}_".format(generate_method)
root_model_save_path = "model_high_mutants/retrain_"
# root_data_path="../data/"
root_data_path="../result/RQ4/"
# G:\code\mutant_generate

model_path=root_model_path+"{}_{}.h5".format(dataset_name,model_name)
# data_path=root_data_path+"{}_{}_{}_test_case.npy".format(dataset_name,model_name,generate_method)
# label_path=root_data_path+"{}_{}_{}_test_case_label.npy".format(dataset_name,model_name,generate_method)
data_path=root_data_path+"{}_{}_minimal_test_set_images.npy".format(dataset_name,model_name)
label_path=root_data_path+"{}_{}_minimal_test_set_labels.npy".format(dataset_name,model_name)


new_data=np.load(data_path,allow_pickle=True)
new_label=np.load(label_path,allow_pickle=True)
new_label=np_utils.to_categorical(new_label,10)

if dataset_name == "mnist":
    x_train, y_train, x_test, y_test = load_mnist()
elif dataset_name == "cifar10":
    x_train, y_train, x_test, y_test = load_cifar10()
elif dataset_name == "fashion_mnist":
    x_train, y_train, x_test, y_test = load_fashion_mnist()

retrain_data=np.append(x_train,new_data,axis=0)
retrain_label=np.append(y_train,new_label,axis=0)
print(retrain_data.shape)

# 随机查看十张图片new_label[idx].argmax(axis=0)
# name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'boat', 'truck']
# index=random.sample(range(new_data.shape[0]),10)
# images=new_data[index]
# fig, ax = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))
# for i, idx in enumerate(index):
#     ax[i].set_axis_off()
#     ax[i].title.set_text(name[np.argmax(new_label[idx])])
#     ax[i].imshow(np.reshape(images[i], (32, 32,3)))
# plt.show()

model=load_model(model_path)
model.summary()

# callbacks只保存测试集准确率最高的模型
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=root_model_save_path+"{}_{}.h5".format(dataset_name,model_name),  # 文件保存路径
    monitor='val_accuracy',                # 监控的指标，例如 'val_accuracy'
    save_best_only=True,                   # 仅保存验证集准确率最高的模型
    mode='max',                            # 监控值的模式：'max' 为取最大值，'min' 为取最小值
    save_weights_only=False,               # 是否只保存模型权重
    verbose=1                              # 显示保存信息
)

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.01), metrics=["accuracy"])
model.fit(
        retrain_data,
        retrain_label,
        epochs=10,
        batch_size=128,
        shuffle=True,
        verbose=1,
        validation_data=(x_test,y_test),
        callbacks=[checkpoint_callback]
    )
# model.save(root_model_save_path+"{}_{}.h5".format(dataset_name,model_name))
print("successfully save the model")