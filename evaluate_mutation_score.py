import random
import tensorflow as tf
import numpy as np
import keras.backend as K
from tensorflow.keras.datasets import mnist,cifar10,fashion_mnist
from keras.models import load_model
from keras.utils import np_utils
import matplotlib.pyplot as plt
from pathlib import Path
from neuronal_analysis import read_file,write_file,load_mnist,load_cifar10,load_fashion_mnist
import os
import gc
# from high_mutant.model_level_operators import operator_name


operator_name = ["GF", "DGF", "WS", "NEB", "NAI", "NS"]
# operator_name = ["DR", "LE", "DM", "DF", "NP"]


def get_random_CAR(score_list):
    max_index=0
    for i in range(len(score_list)):
        if score_list[i]>score_list[max_index]:
            max_index=i
    if score_list[max_index]==0.0:
        return 0
    temp=max([score_list[i] for i in range(len(score_list)) if i!=max_index])
    if temp==0.0:
        return 10
    else:
        return min(10,score_list[max_index]/temp)


def get_mutant_score(error_list):
    category=np.zeros(10)
    for i in error_list:
        category[i]=1
    return sum(category)/10.0


# def get_mutant_score(error_list,predict):
#     category=np.zeros((10,10))
#     for i,j in zip(error_list,predict):
#         category[i][j]=1
#     return np.sum(category)/90.0


def get_error_rate(new_y_predict, correct_label):
    correct = np.zeros(10)
    error = np.zeros(10)
    # 记录每个类的错误率
    for x in range(len(correct_label)):
        if correct_label[x] == new_y_predict[x]:
            correct[correct_label[x]] += 1
        else:
            error[correct_label[x]] += 1
    # print(error/(correct+error))
    return error/(correct+error)


if __name__ == '__main__':
    neuronal_threshold_percentage = 90

    model_name = "lenet5"
    dataset_name = "mnist"
    root_model_path = "F:\code/mutant_generate/train_model/model/model_"
    root_mutant_path = "F:\code/mutant_generate/mutants/"
    root_random_mutant_path = "F:\code/mutant_generate/random_mutants/"

    model_path = "{}{}_{}.h5".format(root_model_path, dataset_name, model_name)
    mutant_path= "{}{}_{}/thre{}_key0.01/".format(root_mutant_path,dataset_name,model_name,neuronal_threshold_percentage)
    random_mutant_path= "{}{}_{}/model_level_mutants/".format(root_random_mutant_path,dataset_name,model_name)

    # mutant_path="{}{}_{}/source_level_mutants/".format(root_mutant_path,dataset_name,model_name)
    # random_mutant_path="{}{}_{}/source_level_mutants/".format(root_random_mutant_path,dataset_name,model_name)

    if not os.path.exists(mutant_path):
        print('There is no path {}'.format(mutant_path))
        exit(1)

    if dataset_name == "mnist":
        x_train, y_train, x_test, y_test = load_mnist()
        # name=['0','1','2','3','4','5','6','7','8','9']
    elif dataset_name == "cifar10":
        x_train, y_train, x_test, y_test = load_cifar10()
        # name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'boat', 'truck']
    elif dataset_name == "fashion_mnist":
        x_train, y_train, x_test, y_test = load_fashion_mnist()


    mut_path=Path(mutant_path)
    # mutants=list(mut_path.glob('mutant_*'))
    ran_mut_path=Path(random_mutant_path)
    random_mutants=list(ran_mut_path.glob('mutant_*'))
    all_error_rate=read_file(mutant_path+"error_rate.txt")

    model=load_model(model_path)

    # random select 1000 test case
    ori_pred_labels = model.predict(x_test).argmax(axis=1)
    correct_index = np.where(ori_pred_labels == y_test.flatten())[0]
    correct_images = x_test[correct_index]
    correct_labels = ori_pred_labels[correct_index]

    # test_index=random.sample(range(correct_labels.shape[0]),1000)
    # test_images=correct_images[test_index]
    # test_labels=correct_labels[test_index]

    # minimal test set
    test_images = np.load("./result/RQ3/{}_{}_minimal_test_set_images.npy".format(dataset_name,model_name))
    test_labels = np.load("./result/RQ3/{}_{}_minimal_test_set_labels.npy".format(dataset_name,model_name))
    print(test_images.shape)


    # mnist:[ 973. 1125. 1013. 1000.  969.  880.  936. 1003.  945.  975.]
    # cifar10:[811. 885. 665. 614. 638. 626. 801. 805. 834. 785.]

    layer_names = [layer.name for layer in model.layers]
    print(mutant_path)
    # print(len(mutants))


    # high_muta_score=read_file(mutant_path+"mutants_score.txt")
    # random_muta_score=read_file(random_mutant_path+"mutants_score.txt")

    mutant_score0=[]
    mutant_score1=[]
    mutant_score2=[]
    mutant_score3=[]
    # error_rate=[]
    car_distribute=[0 for i in range(11)]
    count=0   # 记录未参加的层数
    for i in range(len(layer_names)-1):   # cifar10 vgg16 记得i从12开始
        # break
        if layer_names[i].find('max_pooling2d') >= 0 or layer_names[i].find('flatten') >= 0 or layer_names[i].find('dropout') >= 0:
            count+=1
            continue
        for j in range(0,6):
            if j==1:
                continue
            path=list(mut_path.glob("mutant_{}_{}_category*".format(operator_name[j],layer_names[i])))
            # path = list(mut_path.glob("mutant_{}_category*".format(operator_name[j])))
            for k in range(10):
                K.clear_session()
                mutant=load_model(path[k])
                predict_label = mutant.predict(test_images).argmax(axis=1)
                # temp_err=get_error_rate(predict_label,test_labels)
                # temp_car=get_random_CAR(temp_err)
                # if temp_car==0:
                #     continue
                err=all_error_rate[i-count][j][k]   # error_rate     cifar10 vgg16 记得i-12-count
                # ca=get_CAR(k,err)    # CAR
                car = get_random_CAR(err)  # CAR
                # print(ca)
                if car == 0:
                    # print(path[k])
                    continue
                # if car>=5:
                #     # car_distribute[10]+=1
                #     print(path[k])

                car_distribute[int(car)]+=1
                error_index=np.where(predict_label!=test_labels)[0]
                error_label=test_labels[error_index]
                mut_sc=get_mutant_score(error_label)
                # label = predict_label[error_index]  #
                # mut_sc = get_mutant_score(error_label, label)
                mutant_score0.append(mut_sc)
                if car>=10:
                    mutant_score3.append(mut_sc)
                if car>=5:
                    mutant_score2.append(mut_sc)
                if car>=3:
                    mutant_score1.append(mut_sc)
                del mutant
                gc.collect()
    print("random start")
    print(random_mutant_path)
    # print(len(random_mutants))

    err_rate=[]
    random_mutant_score=[]
    random_mutant_score_car10=[]
    random_all_error_rate=read_file(random_mutant_path+"error_rate.txt")
    random_car_distribute = [0 for i in range(11)]
    count=0
    for path in random_mutants:
        # print(path)
        # continue
        K.clear_session()
        # if str(path).find("_GF_") >= 0 or str(path).find("_NS_") >= 0:
        #     continue
        mutant=load_model(path)
        predict_label = mutant.predict(test_images).argmax(axis=1)
        temp_err = get_error_rate(predict_label, test_labels)  # error_rate
        err_rate.append(temp_err.tolist())

        err=random_all_error_rate[count]
        count+=1
        car = get_random_CAR(err)  # CAR

        if car == 0:
            print(path,'car=0')
            continue
        error_index = np.where(predict_label != test_labels)[0]
        error_label = test_labels[error_index]
        mut_sc = get_mutant_score(error_label)
        if car >= 10:
            # random_car_distribute[10] += 1
            random_mutant_score_car10.append(mut_sc)
            # print(path,car)

        random_car_distribute[int(car)] += 1
        # label = predict_label[error_index]      #
        # mut_sc = get_mutant_score(error_label,label)
        random_mutant_score.append(mut_sc)
        del mutant
        gc.collect()
    # write_file(random_mutant_path+"error_rate.txt",err_rate)


    # print(error_rate)
    plt.title("{}_{}_mutant_score".format(dataset_name,model_name))
    plt.ylabel("Mutant Score")
    plt.ylim(0,1)
    plt.boxplot([random_mutant_score,random_mutant_score_car10,mutant_score0,mutant_score1,mutant_score2,mutant_score3],showmeans=True,showfliers=False)
    plt.xticks(range(1, 7), ["deepmutation", "random_car>10", "car>0", "car>3", "car>5", "car>10"])
    # plt.savefig("./result/{}_{}_mutant_score".format(dataset_name,model_name))
    # plt.savefig("./result/{}_{}_source_mutant_score".format(dataset_name, model_name))
    plt.show()
    print(np.average(random_mutant_score),random_mutant_score)
    print(np.average(random_mutant_score_car10),random_mutant_score_car10)
    print(np.average(mutant_score0),mutant_score0)
    print(np.average(mutant_score1),mutant_score1)
    print(np.average(mutant_score2),mutant_score2)
    print(np.average(mutant_score3),mutant_score3)
    print(car_distribute)
    print(random_car_distribute)

