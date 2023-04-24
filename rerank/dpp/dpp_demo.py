import time
import numpy as np
import math


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def dpp_sw(kernel_matrix, window_size=3, max_length=14, epsilon=1E-10):
    """
    Sliding window version of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param window_size: positive int
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    v = np.zeros((max_length, max_length))
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    window_left_index = 0
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[window_left_index:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        v[k, window_left_index:k] = ci_optimal
        v[k, k] = di_optimal
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[window_left_index:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        if len(selected_items) >= window_size:
            window_left_index += 1
            for ind in range(window_left_index, k + 1):
                t = math.sqrt(v[ind, ind] ** 2 + v[ind, window_left_index - 1] ** 2)
                c = t / v[ind, ind]
                s = v[ind, window_left_index - 1] / v[ind, ind]
                v[ind, ind] = t
                v[ind + 1:k + 1, ind] += s * v[ind + 1:k + 1, window_left_index - 1]
                v[ind + 1:k + 1, ind] /= c
                v[ind + 1:k + 1, window_left_index - 1] *= c
                v[ind + 1:k + 1, window_left_index - 1] -= s * v[ind + 1:k + 1, ind]
                cis[ind, :] += s * cis[window_left_index - 1, :]
                cis[ind, :] /= c
                cis[window_left_index - 1, :] *= c
                cis[window_left_index - 1, :] -= s * cis[ind, :]
            di2s += np.square(cis[window_left_index - 1, :])
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def dpp_model(scores, feature_vectors, item_size, max_length, alpha=0.7, window_size=3):
    """
    DPPModel 最大化相关性和多样性的推荐子集
    :param scores:  候选集排序分数
    :param feature_vectors: 候选集的向量矩阵
    :param item_size: 重排的候选集数量
    :param max_length: 最后选择的候选集数量
    :param alpha: 超参数控制相似性和多样性的关系
    :return:
    """
    alpha = alpha * 1.0 / (2 * (1 - alpha))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    mean = scores.mean()  # 计算平均数
    deviation = scores.std()  # 计算标准差
    # 标准化数据的公式: (数据值 - 平均数) / 标准差
    scores = sigmoid((scores - mean) / deviation)
    # 将数据变换成 0-1 之间
    # scores = sigmoid(scores)

    scores = np.exp(scores * alpha)  # 精排商品的分数
    ones = np.ones((item_size, 1))
    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    feature_vectors = np.concatenate((ones, feature_vectors), axis=1) / np.sqrt(2)

    similarities = np.dot(feature_vectors, feature_vectors.T)  # 相似度矩阵，任意两商品间的距离
    kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))  # 核矩阵的计算方式
    result = dpp(kernel_matrix, max_length)
    # result = dpp_sw(kernel_matrix, window_size, max_length, epsilon=1E-10)
    return result, similarities


if __name__ == '__main__':
    item_size = 15
    feature_dimension = 8
    max_length = 10
    scores= np.random.randn(item_size)
    print(scores)
    feature_vectors = np.random.randn(item_size, feature_dimension)  # 商品的特征向量
    t = time.time()
    result = dpp_model(scores, feature_vectors, item_size, max_length, alpha=0.7, window_size=3)[0]
    # result = dpp_model(scores, feature_vectors, item_size, max_length, alpha=0.7, window_size=3)
    print(result)
    print('algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))