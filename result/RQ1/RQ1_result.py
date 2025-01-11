import numpy as np
import matplotlib.pyplot as plt
import random
from neuronal_analysis import read_file

model_name=["A","B","C","D","E"]
colors=['bisque','white']

def car_distribute():
    car_data=[
        [0, 140, 29, 12, 10, 6, 0, 0, 0, 1, 2],
        [0, 123, 37, 18, 10, 4, 5, 1, 0, 0, 2],
        [0, 180, 34, 11, 3, 4, 1, 2, 1, 0, 11],
        [0, 164, 56, 18, 6, 4, 0, 0, 0, 0, 2],
        [0, 173, 34, 12, 11, 10, 3, 5, 4, 1, 30],
        [0, 203, 39, 24, 7, 4, 8, 3, 3, 1, 8],
        [0, 138, 43, 15, 11, 12, 10, 7, 4, 5, 42],
        [0, 281, 6, 0, 0, 1, 0, 0, 0, 0, 12],
        [0, 183, 46, 20, 20, 14, 6, 1, 1, 1, 8],
        [0, 193, 64, 20, 7, 5, 1, 2, 0, 2, 6]
    ]
    car_group = ["3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10", ">10"]
    # plt.style.use('seaborn-v0_8-pastel')
    # 设置多个子图的布局
    fig, axes = plt.subplots(1, 5, figsize=(15, 6), sharey=True)  # 创建5个金字塔图的子图
    # car_number=read_file("")

    # 绘制每个金字塔图
    for i, ax in enumerate(axes):
        ours = car_data[2*i][3:]
        deepmutation = car_data[2*i+1][3:]


        # 各方向的条形图
        ax.barh(car_group, deepmutation, label="deepmutation", align='center')
        ax.barh(car_group, [-f for f in ours], label="ours", align='center')

        # 添加标题和轴标签
        ax.set_title("Model {}".format(model_name[i]))
        ax.set_yticks(car_group)
        ax.grid(axis='x',linestyle='--',alpha=0.7)

        # 调整轴范围
        max_population = max(max(deepmutation), max(ours))+5
        ax.set_xlim(-max_population, max_population)

        # 显示图例
        if i == 0:
            ax.legend(loc="upper right")
            ax.set_xlabel("Numbers of mutants")

    # 显示图形
    plt.tight_layout()
    plt.savefig()
    plt.show()


def get_mutation_score():
    data = []
    data.append(read_file("mnist_lenet5_mutant_score.txt"))
    data.append(read_file("mnist_unknow_mutant_score.txt"))
    data.append(read_file("cifar10_unknow_mutant_score.txt"))
    data.append(read_file("cifar10_vgg16_mutant_score.txt"))
    data.append(read_file("fashion_mnist_unknow_mutant_score.txt"))
    mut_data=[]
    # plt.style.use('seaborn-v0_8-pastel')
    for i in data:
        for j in i:
            mut_data.append(j)

    plt.figure(figsize=(15, 6))
    plt.title("model A                               model B                               model C                               model D                               model E")
    plt.ylabel("Mutant Score")
    plt.ylim(-0.1, 1.1)
    box=plt.boxplot(mut_data,showmeans=True,showfliers=False,patch_artist=True)
    plt.xticks(range(1, 11),
               ["deepmutation", "ours", "deepmutation", "ours", "deepmutation", "ours", "deepmutation", "ours",
                "deepmutation", "ours"])

    for patch,color in zip(box['boxes'],5*colors):
        patch.set_facecolor(color)
    plt.show()
    plt.savefig("")

if __name__ == "__main__":
    car_distribute()
    # get_mutation_score()