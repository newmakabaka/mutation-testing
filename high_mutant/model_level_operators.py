from tensorflow.keras.models import Sequential
import numpy as np
import random

# GF:权值模糊,WS:打乱权值,NEB:神经元传递阻塞,NAI:激活值取反,NS:交换同一层的两个神经元
operator_name = ["GF", "DGF", "WS", "NEB", "NAI", "NS"]


def model_level_operators(model, operator, neuron_list, layer_name):
    layer_weight = model.get_layer(layer_name).get_weights()

    if operator == 0:
        new_weight = gaussian_fuzzing(layer_weight, neuron_list)
    elif operator == 1:
        new_weight = decrease_gaussian_fuzzing(layer_weight, neuron_list, 0, 0.5)
    elif operator == 2:
        new_weight = weights_shuffle(layer_weight, neuron_list)
    elif operator == 3:
        new_weight = neuron_effect_blocking(layer_weight, neuron_list)
    elif operator == 4:
        new_weight = neuron_activation_inverse(layer_weight, neuron_list)
    elif operator == 5:
        new_weight = neuron_switch(layer_weight, neuron_list)

    model.get_layer(layer_name).set_weights(new_weight)
    return model


def gaussian_fuzzing(weight_bias, neurons, standard_deviation=0.5):
    """
    对某一层选定的神经元的权重进行高斯模糊
    :param weight_bias: 某一层的权重和偏移量
    :param neurons: 选定的神经元id
    :param standard_deviation: 标准差
    :return: 高斯模糊后的新权重
    """
    weight_bias = weight_bias.copy()
    weight = weight_bias[0].T
    weight_shape = weight[0].shape
    if len(weight_shape) == 3:
        for i in neurons:
            flatten_weights = weight[i].flatten()
            for j in range(len(flatten_weights)):
                fuzz = np.random.normal(loc=0.0, scale=standard_deviation, size=None)
                flatten_weights[j] *= 1+fuzz
            weight[i] = flatten_weights.reshape(weight_shape)
            # print(weight[i])
    else:
        for i in neurons:
            for j in range(len(weight[i])):
                fuzz = np.random.normal(loc=0.0, scale=standard_deviation, size=None)
                weight[i][j] *= 1 + fuzz
    weight_bias[0] = weight.T

    return weight_bias


def decrease_gaussian_fuzzing(weight_bias, neurons, small, big):
    weight_bias = weight_bias.copy()
    weight = weight_bias[0].T
    weight_shape = weight[0].shape
    if len(weight_shape) == 3:
        for i in neurons:
            flatten_weights = weight[i].flatten()
            for j in range(len(flatten_weights)):
                flatten_weights[j] *= random.uniform(small, big)
            weight[i] = flatten_weights.reshape(weight_shape)
            # print(weight[i])
    else:
        for i in neurons:
            for j in range(len(weight[i])):
                weight[i][j] *= random.uniform(small, big)
    weight_bias[0] = weight.T

    return weight_bias


def weights_shuffle(weight_bias, neurons):
    """
    将某一层选定的神经元的权值打乱
    :param weight_bias: 某一层的权重和偏移量
    :param neurons: 选定的神经元id
    :return: 打乱权值后的新权重
    """
    weight_bias = weight_bias.copy()
    weight = weight_bias[0].T
    weight_shape = weight[0].shape
    if len(weight_shape) == 3:
        for i in neurons:
            flatten_weights = weight[i].flatten()
            np.random.shuffle(flatten_weights)
            weight[i] = flatten_weights.reshape(weight_shape)
            # print(weight[i])
    else:
        for i in neurons:
            np.random.shuffle(weight[i])
    weight_bias[0] = weight.T
    return weight_bias


def neuron_effect_blocking(weight_bias, neurons):
    """
    将某一层选定的神经元的权值变为0
    :param weight_bias: 某一层的权重和偏移量
    :param neurons: 选定的神经元id
    :return: 阻断后的新权值
    """
    weight_bias = weight_bias.copy()
    weight = weight_bias[0].T
    weight_shape = weight[0].shape
    for i in neurons:
        weight[i] = np.zeros(weight_shape)
    weight_bias[0] = weight.T
    return weight_bias


def neuron_activation_inverse(weight_bias, neurons):
    """
    将某一层选定的神经元的权值反转
    :param weight_bias: 某一层的权重和偏移量
    :param neurons: 选定的神经元id
    :return: 反转后的新权值
    """
    weight_bias = weight_bias.copy()
    weight = weight_bias[0].T
    for i in neurons:
        weight[i] = -weight[i]
    weight_bias[0] = weight.T
    return weight_bias


def neuron_switch(weight_bias, neurons):
    """
    将某一层的神经元权值随机进行交换
    :param weight_bias:
    :param neurons:
    :return:
    """
    weight_bias = weight_bias.copy()
    weight = weight_bias[0].T
    shuffled_neurons = neurons.copy()

    for i in shuffled_neurons:
        index=random.choice(range(weight.shape[0]))
        while index==i:
            index=random.choice(range(weight.shape[0]))
        temp=weight[index].copy()
        weight[index]=weight[i]
        weight[i]=temp
    weight_bias[0] = weight.T
    return weight_bias


# def neuron_switch(weight_bias, neurons):
#     """
#     将某一层的神经元权值随机进行交换
#     :param weight_bias:
#     :param neurons:
#     :return:
#     """
#     weight_bias = weight_bias.copy()
#     weight = weight_bias[0].T
#     shuffled_neurons = neurons.copy()
#     random.shuffle(shuffled_neurons)
#     temp = []
#     for i in shuffled_neurons:
#         temp.append(weight[i])
#     for j in range(len(neurons)):
#         weight[neurons[j]] = temp[j]
#     weight_bias[0] = weight.T
#     return weight_bias