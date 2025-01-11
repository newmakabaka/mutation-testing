import numpy as np
from keras.models import load_model
import tensorflow as tf
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import ast
from tqdm import tqdm
from tensorflow.keras.datasets import mnist,cifar10,fashion_mnist

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


def get_matrx(layer_out):
    if layer_out[0].ndim == 3:
        temp = []
        for i in range(len(layer_out)):
            temp.append(_aggr_output(layer_out[i]))
        layer_matrx = np.array(temp)
    else:
        layer_matrx = np.array(layer_out)

    # neuronal_threshold=np.percentile(layer_matrx,neuronal_threshold_percentage,axis=0)
    # neuronal_threshold=neuronal_threshold.tolist()
    # print('-------------------------')
    return layer_matrx


def get_neuronal_output(model, analysed_layers, slice_count, train, path):
    print("start analyse")
    # neuronal_threshold = []      # 每一层每个神经元的阈值
    for layer in analysed_layers:
        model_layer_outputs = [layer.output]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=model_layer_outputs)
        layer_matrix = None      # 该层的输出
        for i in range(slice_count):
            if (i+1)*dataset_slice > len(train):
                train_slice = train[i*dataset_slice:]
            else:
                train_slice = train[i*dataset_slice:(i+1)*dataset_slice]
            activations = activation_model.predict(train_slice, batch_size=batch_size)
            # neuronal_thresholds=[]
            temp_matrix = get_matrx(activations)
            if layer_matrix is None:
                layer_matrix = temp_matrix
            else:
                layer_matrix = np.append(layer_matrix, temp_matrix, axis=0)

        print(layer.name+"-layer analysis completed")
        output_path = path+layer.name+".npy"
        np.save(output_path, layer_matrix)


def get_neuronal_threshold(layers, path):
    threshold = []
    for l in layers:
        out = np.load(path + l.name + ".npy")
        temp = np.percentile(out, neuronal_threshold_percentage, axis=0)
        print(temp.shape)
        threshold.append(temp.tolist())
    write_file(path+"threshold"+str(neuronal_threshold_percentage)+".txt", threshold)
    return threshold


def get_key_neurons(path,layers,threshold):
    i = 0  # 表示第几层
    key_neurons = []  # 每一层，每一类所对应的关键神经元
    if not os.path.exists(path + "figure/"):
        os.makedirs(path + "figure/")

    for layer in layers:
        neuronal_idx = range(layer.output_shape[-1])  # 柱状图的x坐标，也是神经元的编号
        # activate_neuronal_num=[[0 for i in range(layer.output_shape[-1])]for j in range(10)]     # 柱状图的y坐标，也是该层每个神经元被激活的次数
        activate_neuronal_num = np.zeros(shape=(10, layer.output_shape[-1]))  # 柱状图的y坐标，也是该层每个神经元被激活的次数
        out_path = path + layer.name + ".npy"
        out = np.load(out_path)
        for j in tqdm(range(out.shape[0])):  # 每个输入
            for idx in neuronal_idx:  # 每个神经元
                if out[j][idx] > threshold[i][idx]:
                    activate_neuronal_num[correct_labels[j]][idx] += 1
        key_neurons_num = max(1, int(activate_neuronal_num.shape[-1] * key_neurons_percentage))  # 关键神经元的个数
        temp_neurons = None
        for k in range(10):
            # plt.title("Activation of neurons in class "+str(k)+" of layer "+layer.name)
            # plt.xlabel("neuronal_id")
            # plt.ylabel("activate_neuronal_num")
            # plt.bar(neuronal_idx,activate_neuronal_num[k],color=colors[i])
            # plt.savefig(path+"figure/class "+str(k)+" of layer "+layer.name)
            # plt.clf()

            neurons = np.argsort(-activate_neuronal_num[k])[:key_neurons_num]  # 激活次数最高的n个神经元的索引，也就是神经元的idx
            if temp_neurons is None:
                temp_neurons = neurons.reshape((1, key_neurons_num))
            else:
                temp_neurons = np.append(temp_neurons, neurons.reshape((1, key_neurons_num)), axis=0)
        key_neurons.append(temp_neurons.tolist())
        i += 1

    write_file(path + "threshold" + str(neuronal_threshold_percentage) + "_key_neurons" + str(
        key_neurons_percentage) + ".txt", key_neurons)
    return key_neurons


def write_file(path,data):
    with open(path,'w') as f:
        for t in data:
            f.write(str(t))
            f.write('\n')
    print("successfully save the file")


def read_file(path):
    data = []
    with open(path,'r') as f:
        for line in f:
            data.append(ast.literal_eval(line))
    return data


def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    test_images = test_images.astype('float32') / 255.0
    return train_images,train_labels,test_images,test_labels


def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.reshape((train_images.shape[0],32,32,3))
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.reshape((test_images.shape[0], 32, 32, 3))
    test_images = test_images.astype('float32') / 255.0
    return train_images, train_labels, test_images, test_labels


def load_fashion_mnist():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    test_images = test_images.astype('float32') / 255.0
    return train_images,train_labels,test_images,test_labels


if __name__ == "__main__":
    # file path
    model_path = "./train_model/model/model_mnist_lenet5.h5"
    root_path = "./layer_output/mnist_lenet5/"

    # parameter
    neuronal_threshold_percentage = 90
    batch_size = 500
    dataset_slice = 5000
    key_neurons_percentage = 0.01

    model = load_model(model_path)
    model.summary()

    # 加载数据集
    if root_path.find("fashion_mnist") >= 0:
        x_train, y_train, x_test, y_test = load_fashion_mnist()
    elif root_path.find("cifar10") >= 0:
        x_train, y_train, x_test, y_test = load_cifar10()
    elif root_path.find("mnist") >= 0:
        x_train, y_train, x_test, y_test = load_mnist()

    # 过滤出正确的结果
    # train
    ori_pred_labels = model.predict(x_train).argmax(axis=1)
    correct_index = np.where(ori_pred_labels == y_train.flatten())[0]
    correct_images = x_train[correct_index]
    correct_labels = ori_pred_labels[correct_index]      # 显示的是数字而不是向量即 5 而不是[0,0,0,0,0,1,0,0,0,0]

    # test
    # ori_pred_labels = model.predict(test_images).argmax(axis=1)
    # correct_index = np.where(ori_pred_labels == test_labels.flatten())[0]
    # correct_images = test_images[correct_index]
    # correct_labels = ori_pred_labels[correct_index]

    analyse_layers = model.layers[:-1]  # 不考虑输出层
    sl_count = int(len(correct_images) / dataset_slice) + 1

    # 获得每个测试输入，每一层，每个神经元的输出
    if not os.path.exists(root_path):
        os.makedirs(root_path)
        get_neuronal_output(model,analyse_layers,sl_count,correct_images,root_path)

    # 获得每一层的神经元激活阈值
    thre_path = root_path+"threshold"+str(neuronal_threshold_percentage)+".txt"
    if os.path.exists(thre_path):
        print("find exist threshold file")
        neuronal_threshold = read_file(thre_path)
    else:
        neuronal_threshold = get_neuronal_threshold(analyse_layers,root_path)

    # 获得每个层，每个类的关键神经元
    key_neur_path = root_path + "threshold" + str(neuronal_threshold_percentage) + "_key_neurons" + str(key_neurons_percentage) + ".txt"
    if os.path.exists(key_neur_path):
        print("find exist key neurons file")
        key_neurons = read_file(key_neur_path)
    else:
        key_neurons = get_key_neurons(root_path,analyse_layers,neuronal_threshold)

    save_path="{}figure/thre{}_key{}/".format(root_path,neuronal_threshold_percentage,key_neurons_percentage)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    category = [i for i in range(10)]
    for l in range(len(key_neurons)):     # 每个层
        # print(key_neurons[l])
        # print(len(key_neurons[l][0]))
        plt.figure(figsize=(15,9))
        exist_count=[[0.0 for b in range(10)]for c in range(10)]
        for i in range(10):   # 每个类
            for j in range(10):
                if i==j:
                    continue
                else:
                    exist_count[i][j] += len(set(key_neurons[l][i]) & set(key_neurons[l][j]))/len(key_neurons[l][0])

            plt.subplot(2,5,i+1)
            plt.bar(category, exist_count[i], color=colors[i])
            plt.title("category "+str(i))
            plt.xlabel("class")
            if i%5==0:
                plt.ylabel("percentage")
            plt.ylim(0,1)
        # plt.show()
        plt.savefig(save_path+"layer "+analyse_layers[l].name)
        plt.close()
        print("layer {} finsh".format(analyse_layers[l].name))


# (9855, 28, 28, 32)
# (9855, 28, 28, 32)
# (9855, 14, 14, 32)
# (9855, 14, 14, 64)
# (9855, 14, 14, 64)
# (9855, 7, 7, 64)
# (9855, 3136)
# (9855, 200)
# (9855, 10)