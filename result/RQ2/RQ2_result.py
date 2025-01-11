import numpy as np
import matplotlib.pyplot as plt
import random
from neuronal_analysis import read_file
import seaborn as sns

model_name=["A","B","C","D","E"]

def ablation():
    data=[]
    data.append(read_file("mnist_lenet5_abl.txt"))
    data.append(read_file("mnist_unknow_abl.txt"))
    data.append(read_file("cifar10_unknow_abl.txt"))
    data.append(read_file("cifar10_vgg16_abl.txt"))
    data.append(read_file("fashion_mnist_unknow_abl.txt"))

    for i in range(len(data)):
        # plt.title("Model {}".format(model_name[i]))
        plt.ylabel("Mutant Score")
        plt.ylim(-0.1, 1.1)
        plt.boxplot(data[i],showmeans=True,showfliers=False)
        plt.xticks(range(1,5),["None", "CAR", "neurons", "CAR+neurons"])
        # plt.show()
        plt.savefig("model{}_abl.png".format(model_name[i]))
        plt.close()


def alt_car():
    data = []
    data.append(read_file("mnist_lenet5_altcar.txt"))
    data.append(read_file("mnist_unknow_altcar.txt"))
    data.append(read_file("cifar10_unknow_altcar.txt"))
    data.append(read_file("cifar10_vgg16_altcar.txt"))
    data.append(read_file("fashion_mnist_unknow_altcar.txt"))
    for i in range(len(data)):
        # plt.title("Model {}".format(model_name[i]))
        plt.ylabel("Mutant Score")
        plt.ylim(-0.1, 1.1)
        plt.boxplot(data[i],showmeans=True,showfliers=False,patch_artist=True,boxprops={'facecolor':'bisque'})
        plt.xticks(range(1,5),["CAR>=0", "CAR>=3", "CAR>=5", "CAR>=10"])
        # plt.show()
        plt.savefig("model{}_altcar.png".format(model_name[i]))
        plt.close()

if __name__ == "__main__":
    ablation()
    # alt_car()