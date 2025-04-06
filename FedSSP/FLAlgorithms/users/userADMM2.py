import torch
import os
import json
from FLAlgorithms.users.userbase import User
import numpy as np
import copy

'''Implementation for FedPCA clients'''

class UserADMM2():
    def __init__(self, algorithm, device, id, train_data, commonPCA, learning_rate, ro, local_epochs, dim):
        # 初始化客户端的参数和数据
        self.localPCA = copy.deepcopy(commonPCA)  # 深拷贝公共主成分分析矩阵作为本地PCA矩阵 (U)
        self.localZ = copy.deepcopy(commonPCA)   # 深拷贝公共主成分分析矩阵作为本地Z矩阵
        self.localY = copy.deepcopy(commonPCA)   # 深拷贝公共主成分分析矩阵作为本地Y矩阵（对偶变量）
        self.localT = torch.matmul(self.localPCA.T, self.localPCA)  # 计算本地T矩阵（用于约束条件）
        self.ro = ro  
        self.device = device  # 使用的设备（CPU或GPU）
        self.id = id  # 客户端ID
        self.train_samples = len(train_data)  # 本地训练样本数量
        self.learning_rate = learning_rate  # 学习率
        self.local_epochs = local_epochs  # 本地训练轮数
        self.dim = dim  # 降维后的维度
        self.train_data = train_data.T  # 转置训练数据，使其形状为 [dxN]，其中d是特征数，N是样本数
        self.algorithm = algorithm  # 算法名称（FedPE 或 FedPG）
        self.localPCA.requires_grad_(True)  # 启用梯度计算

    def set_commonPCA(self, commonPCA):
        # 更新本地Z矩阵和Y矩阵（对偶变量）
        self.localZ = commonPCA.data.clone()  # 将公共PCA矩阵复制到本地Z矩阵
        self.localY = self.localY + self.ro * (self.localPCA - self.localZ)  # 更新本地Y矩阵
        
        # 更新本地T矩阵（用于约束条件）
        temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
        hU = torch.max(torch.zeros(temp.shape), temp)**2  # 计算h(U)函数
        self.localT = self.localT + self.ro * hU  # 更新本地T矩阵

    def train_error_and_loss(self):
        # 计算训练误差和损失
        residual = torch.matmul((torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)), self.train_data)
        loss_train = torch.norm(residual, p="fro") ** 2 / self.train_samples  # Frobenius范数平方除以样本数
        return loss_train, self.train_samples

    def hMax(self):
        # 计算h(U)的最大值
        temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
        return torch.max(torch.zeros(temp.shape), temp)

    def train(self, epochs):
        for i in range(self.local_epochs):
            if self.algorithm == "FedPE":
                '''Euclidean space'''
                # 在欧几里得空间中进行训练
                self.localPCA.requires_grad_(True)  # 启用梯度计算
                residual = torch.matmul(torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T), self.train_data)
                temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
                hU = torch.max(torch.zeros(temp.shape), temp)**2  # 计算h(U)函数
                
                # 计算正则化项和Frobenius内积项     正则化 是否存在？？？？
                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ)**2 + 0.5 * self.ro * torch.norm(hU) ** 2
                frobenius_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ)) + torch.sum(torch.inner(self.localT, hU))
                
                # 计算总损失
                self.loss = 1/self.train_samples * torch.norm(residual, p="fro") ** 2
                self.lossADMM = self.loss + 1/self.train_samples * (frobenius_inner + regularization)
                
                # 保存当前的localPCA副本
                temp = self.localPCA.data.clone()
                
                # 清空梯度
                if self.localPCA.grad is not None:
                    self.localPCA.grad.data.zero_()
                
                # 反向传播计算梯度
                self.lossADMM.backward(retain_graph=True)
                
                # 更新localPCA
                temp = temp - self.learning_rate * self.localPCA.grad
                self.localPCA = temp.data.clone()
                 
            else:
                '''Grassmannian manifold'''
                # 在格拉斯曼流形上进行训练
                self.localPCA.requires_grad_(True)  # 启用梯度计算
                residual = torch.matmul(torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T), self.train_data)
                frobenius_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ))
                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ)**2
                
                # 计算总损失
                self.loss = 1/self.train_samples * torch.norm(residual, p="fro") ** 2
                self.lossADMM = self.loss + 1/self.train_samples * (frobenius_inner + regularization)
                
                # 保存当前的localPCA副本
                temp = self.localPCA.data.clone()
                
                # 清空梯度
                if self.localPCA.grad is not None:
                    self.localPCA.grad.data.zero_()
                
                # 反向传播计算梯度
                self.lossADMM.backward(retain_graph=True)
                
                '''Moving on Grassmannian manifold'''
                # 投影到切空间
                projection_matrix = torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)
                projection_gradient = torch.matmul(projection_matrix, self.localPCA.grad)
                temp = temp - self.learning_rate * projection_gradient
                
                # 通过QR分解将结果映射回格拉斯曼流形
                q, r = torch.linalg.qr(temp)
                self.localPCA = q.data.clone()
                # print("W:", np.linalg.norm(self.localPCA, ord=2))

        return