import numpy as np
import torch

def get_adj(path):
    '''
    读取adj矩阵
    会跳过第一行
    '''
    adj=np.loadtxt(path,delimiter=',',skiprows=1) 
    adj=torch.from_numpy(adj).to(torch.float32) 
    return adj

def distance_matrix(adj_matrix):
    '''
    根据邻接矩阵计算距离矩阵
    '''
    n = len(adj_matrix)
    distance_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i][j] = 0
            elif adj_matrix[i][j] == 1:
                distance_matrix[i][j] = 1
            else:
                distance_matrix[i][j] = float('inf')
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = min(distance_matrix[i][j], distance_matrix[i][k] + distance_matrix[k][j])
    return distance_matrix

def k_adj_matrix(distance_matrix, k):
    '''
    根据距离矩阵计算k阶邻接矩阵
    注意: 仅保留距离恰好为k时的节点
    '''
    n = len(distance_matrix)
    adj_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if distance_matrix[i][j] == k:  # 只有第i个点和第j个点之间距离恰好为k时，邻接矩阵中对应位置才会赋值为1
                adj_matrix[i][j] = 1
    adj_matrix=torch.from_numpy(adj_matrix).to(torch.float32)
    return adj_matrix


if __name__=='__main__':
    adj=get_adj(path='./data/adjacency_with_label.csv')
    distance=distance_matrix(adj)
    k_adj=k_adj_matrix(distance , 28)
    print(k_adj)



