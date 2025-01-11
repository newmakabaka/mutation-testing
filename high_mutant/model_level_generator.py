import tensorflow as tf
from keras.models import load_model
from model_level_operators import *
from neuronal_analysis import read_file, load_mnist, load_cifar10, load_fashion_mnist, colors, write_file
import os
import matplotlib.pyplot as plt
from evaluate_mutation_score import get_CAR,get_mutant_score,get_error_rate,get_random_CAR


model_name = "vgg16"
dataset_name = "cifar10"

root_model_path = "../train_model/model/model_"
root_neurons_path = "../layer_output/"
root_model_save_path = "G:/code/mutant_generate/mutants/"

neuronal_threshold_percentage = 90


if __name__ == '__main__':
    model_path = "{}{}_{}.h5".format(root_model_path, dataset_name, model_name)
    neurons_path = "{}{}_{}/threshold{}_key_neurons0.05.txt".format(root_neurons_path, dataset_name, model_name,neuronal_threshold_percentage)
    model_save_path = "{}{}_{}/thre{}_key0.05/".format(root_model_save_path, dataset_name, model_name,neuronal_threshold_percentage)

    model = load_model(model_path)
    model.summary()

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

    neurons = read_file(neurons_path)
    layer_names = [layer.name for layer in model.layers]
    print(layer_names)

    score=[]
    #len(layer_names)-1
    for i in range(12,len(layer_names)-1):     # 每一层
        if layer_names[i].find('max_pooling2d') >= 0 or layer_names[i].find('flatten') >= 0 or layer_names[i].find('dropout') >= 0:
            continue
        print('layer {} start'.format(layer_names[i]))
        each_op=[]
        #len(operator_name)
        for j in range(0,6):   # 每个变异算子
            print('operator {} start'.format(operator_name[j]))
            plt.figure(figsize=(15, 9))
            each_class=[]
            for k in range(10):    # 每个类
                model = load_model(model_path)
                new_model = model_level_operators(model, j, neurons[i][k], layer_names[i])     # 关键神经元
                new_y_predict = new_model.predict(correct_images).argmax(axis=1)
                error_rate = get_error_rate(new_y_predict, correct_labels)
                # car=get_CAR(k,error_rate)

                new_model.save("{}mutant_{}_{}_category{}.h5".format(model_save_path, operator_name[j], layer_names[i], k))

                del model
                del new_model
                each_class.append(error_rate.tolist())
                plt.subplot(2, 5, k + 1)
                plt.bar([i for i in range(10)], error_rate, color=colors[i%len(colors)])
                plt.title("category " + str(k))
                plt.xlabel("class")
                if k % 5 == 0:
                    plt.ylabel("killed percentage")
            each_op.append(each_class)
            # plt.show()
            plt.savefig("{}operator_{}_layer_{}".format(model_save_path, operator_name[j], layer_names[i]))
            plt.close()
            print('operator {} finished'.format(operator_name[j]))
        print('layer {} finished'.format(layer_names[i]))
        score.append(each_op)
    write_file(model_save_path+"error_rate.txt", score)