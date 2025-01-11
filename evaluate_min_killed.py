import random
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from pathlib import Path
from neuronal_analysis import read_file,load_mnist,load_cifar10,load_fashion_mnist
import keras.backend as K
import gc


if __name__ == '__main__':
    neuronal_threshold_percentage = 90

    model_name = "lenet5"
    dataset_name = "mnist"
    root_model_path = "F:\code/mutant_generate/train_model/model/model_"
    root_mutant_path = "F:\code/mutant_generate/mutants/"
    root_random_mutant_path = "F:\code/mutant_generate/random_mutants/"

    model_path = "{}{}_{}.h5".format(root_model_path, dataset_name, model_name)
    mutant_path= "{}{}_{}/selected_mutants/".format(root_mutant_path,dataset_name,model_name)
    random_mutant_path= "{}{}_{}/model_level_mutants/".format(root_random_mutant_path,dataset_name,model_name)

    # mutant_path="{}{}_{}/source_level_mutants/".format(root_mutant_path,dataset_name,model_name)
    # random_mutant_path="{}{}_{}/source_level_mutants/".format(root_random_mutant_path,dataset_name,model_name)

    if dataset_name == "mnist":
        x_train, y_train, x_test, y_test = load_mnist()
        # name=['0','1','2','3','4','5','6','7','8','9']
    elif dataset_name == "cifar10":
        x_train, y_train, x_test, y_test = load_cifar10()
        # name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'boat', 'truck']
    elif dataset_name == "fashion_mnist":
        x_train, y_train, x_test, y_test = load_fashion_mnist()

    model = load_model(model_path)
    ori_pred_labels = model.predict(x_test).argmax(axis=1)
    correct_index = np.where(ori_pred_labels == y_test.flatten())[0]
    correct_images = x_test[correct_index]
    correct_labels = ori_pred_labels[correct_index]

    mut_path = Path(mutant_path)
    mutants = list(mut_path.glob('mutant_*'))
    ran_mut_path = Path(random_mutant_path)
    random_mutants = list(ran_mut_path.glob('mutant_*'))
    print(len(mutants))
    print(len(random_mutants))

    all_killed_number = []
    random_all_killed_number = []
    for i in range(5):
        shuffle_index = [i for i in range(correct_images.shape[0])]
        random.shuffle(shuffle_index)
        # print(shuffle_index)
        shuffle_images = correct_images[shuffle_index]
        shuffle_labels = correct_labels[shuffle_index]
        print("start {} evaluation".format(i+1))
        count=[]
        ran_count=[]
        for path in mutants:
            K.clear_session()
            mutant = load_model(path)
            predict_label = mutant.predict(shuffle_images).argmax(axis=1)
            index=np.where(predict_label != shuffle_labels)
            if len(index[0])!=0:
                count.append(index[0][0])
            del mutant
            gc.collect()

        for path in random_mutants:
            K.clear_session()
            mutant = load_model(path)
            predict_label = mutant.predict(shuffle_images).argmax(axis=1)
            index = np.where(predict_label != shuffle_labels)
            # if len(index[0])<=10 and len(index[0])!=0:
            #     print(path)
            if len(index[0])!=0:
                ran_count.append(index[0][0])
            del mutant
            gc.collect()

        print(max(count))
        print(max(ran_count))
        all_killed_number.append(max(count))
        random_all_killed_number.append(max(ran_count))

    print(all_killed_number)
    print(random_all_killed_number)