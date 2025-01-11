import tensorflow as tf
from keras.models import load_model
from model_level_operators import *
from neuronal_analysis import read_file, load_mnist, load_cifar10, load_fashion_mnist, colors, write_file
import os
import matplotlib.pyplot as plt


def get_CAR(index,score_list):
    score=score_list[index]

    temp=max([score_list[i] for i in range(len(score_list)) if i != index])
    if score==0.0:
        return 0
    if temp==0.0:
        return 10
    else:
        return min(10,score/temp)

if __name__ == '__main__':
    model_name = "unknow"
    dataset_name = "fashion_mnist"

    root_model_path = "../train_model/model/model_"
    root_neurons_path = "../layer_output/"
    root_model_save_path = "../mutants/"
    root_ran_model_save_path = "../random_mutants/"

    neuronal_threshold_percentage = 90


    model_path = "{}{}_{}.h5".format(root_model_path, dataset_name, model_name)
    # neurons_path = "{}{}_{}/threshold{}_key_neurons0.1.txt".format(root_neurons_path, dataset_name, model_name,neuronal_threshold_percentage)
    model_save_path = "{}{}_{}/thre{}_key0.1/".format(root_model_save_path, dataset_name, model_name,neuronal_threshold_percentage)
    ran_model_save_path="{}{}_{}/thre{}_key0.1/".format(root_ran_model_save_path, dataset_name, model_name,neuronal_threshold_percentage)
    neurons_path = "{}{}_{}/threshold{}_key_neurons0.1.txt".format(root_neurons_path, dataset_name, model_name,neuronal_threshold_percentage)

    model = load_model(model_path)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if dataset_name == "mnist":
        x_train, y_train, x_test, y_test = load_mnist()
    elif dataset_name == "cifar10":
        x_train, y_train, x_test, y_test = load_cifar10()
    elif dataset_name == "fashion_mnist":
        x_train, y_train, x_test, y_test = load_fashion_mnist()

    ori_pred_labels = model.predict(x_train).argmax(axis=1)
    correct_index = np.where(ori_pred_labels == y_train.flatten())[0]
    correct_images = x_train[correct_index]
    correct_labels = ori_pred_labels[correct_index]  # 显示的是数字而不是向量即 5 而不是[0,0,0,0,0,1,0,0,0,0]

    score = read_file(model_save_path+"mutants_score.txt")
    layer_names = [layer.name for layer in model.layers if (layer.name.find('max_pooling2d') < 0 and layer.name.find('flatten') < 0 and layer.name.find('dropout') < 0)]
    print(layer_names)
    # 'max_pooling2d' 'flatten' 'dropout'
    neurons = read_file(neurons_path)

    count=0
    for i,sc in zip(range(len(layer_names))[:-1],score):
        print(layer_names[i])
        for name,op in zip(range(6),sc):
            # print(name)
            print(operator_name[name])
            for j in range(len(op)):
                high=get_CAR(j,op[j])
                if high>=3:
                    count+=1
                    print(round(high,2),end=' ')
                    # neurons_num = model.get_layer(layer_names[i]).get_weights()[0].shape[-1]
                    # random_neurons = random.sample(range(neurons_num), len(neurons[i][0]))
                    # new_model = model_level_operators(model, name, random_neurons, layer_names[i])
                    # new_model.save(ran_model_save_path+"mutant_{}_{}_category{}.h5".format(operator_name[name],layer_names[i],j))
                else:
                    print('---',end=' ')
            print(' ')
    print(count)


# for i in range(len(layer_names)-1):
#     if i<=13:
#         continue
#     if layer_names[i].find('max_pooling2d') >= 0 or layer_names[i].find('flatten') >= 0 or layer_names[i].find(
#             'dropout') >= 0:
#         continue
#     print('layer {} start'.format(layer_names[i]))
#     for j in range(1,5):   # 变异算子
#         print('operator {} start'.format(operator_name[j]))
#         for k in range(10):    # 每个类
#             model = load_model(model_path)
                # neurons_num = model.get_layer(layer_names[i]).get_weights()[0].shape[-1]
                # random_neurons = random.sample(range(neurons_num), len(neurons[i][0]))
                # new_model = model_level_operators(model, j, random_neurons, layer_names[i])      # 随机选择的神经元
#             new_model = model_level_operators(model, j, neurons[i][k], layer_names[i])     # 关键神经元
#             new_model.save(model_save_path+"mutant_{}_{}_category{}.h5".format(operator_name[j],layer_names[i],k))