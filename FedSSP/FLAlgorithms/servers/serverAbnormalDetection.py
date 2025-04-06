import torch
import os

from FLAlgorithms.users.userADMM import UserADMM
from FLAlgorithms.users.userADMM2 import UserADMM2
from FLAlgorithms.servers.serverbase2 import Server2
from utils.store_utils import metrics_exp_store
from utils.test_utils import unsw_nb15_test, nsl_kdd_test, iot23_test, ton_test
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

''' Implementation for FedPCA Server'''

class AbnormalDetection(Server2):
    def __init__(self, algorithm, experiment, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, clients, num_users, dim, time, exp_type):
        # 调用父类的初始化方法
        super().__init__(device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time)

        # 初始化参数
        self.algorithm = algorithm  # 算法名称
        self.local_epochs = local_epochs  # 每个客户端本地训练轮数
        self.dataset = dataset  # 数据集名称
        self.num_clients = clients  # 客户端数量
        self.experiment = experiment  # 实验对象（可能是用于记录实验数据的对象）
        self.experiment_type = exp_type  # 实验类型

        # 根据数据集加载相应的数据，并计算每个客户端的数据量
        if self.dataset == "Unsw":
            dataX = self.get_data_unsw_nb15()
            factor = dataX.shape[0] / self.num_clients  # UNSW NB15 数据集总样本数为 56000
        elif self.dataset == "Iot23":
            dataX = self.get_data_Iot23()
            factor = dataX.shape[0] / self.num_clients  # IoT23 数据集总样本数为 29539
        elif self.dataset == "Ton":
            dataX = self.get_data_Ton()
            factor = dataX.shape[0] / self.num_clients  # ToN IoT 数据集总样本数为 81872
        else:
            dataX = self.get_data_snl_kdd()
            factor = dataX.shape[0] / self.num_clients  # NSL KDD 数据集总样本数为 67340
        
        print(f"Total number of training samples: {dataX.shape[0]}")
        
        self.learning_rate = learning_rate  # 学习率
        self.user_fraction = num_users  # 参与每轮全局训练的用户比例
        total_users = self.num_clients
        print("total users: ", total_users)
        
        # 初始化客户端并分配数据          local rounds
        for i in range(self.num_clients):            
            id = i
            train = self.get_client_data(dataX, factor=factor, i=i)
            train = torch.Tensor(train)
            if i == 0:
                _, _, U = torch.svd(train)  # 获取U矩阵的维度
                U = U[:, :dim]
                self.commonPCAz = torch.rand_like(U, dtype=torch.float) # 初始化U矩阵，随机选取
                
            user = UserADMM2(algorithm, device, id, train, self.commonPCAz, learning_rate, ro, local_epochs, dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Selected user in each Global Iteration / Total users:", int(num_users * total_users), " / ", total_users)
        print("-------------------Finished creating FedPCA server-------------------")

    '''
    从CSV文件中获取数据
    '''
    def get_data(self, i):
        directory = os.getcwd()  # 获取当前工作目录
        data_path = os.path.join(directory, "abnormal_detection_data/train")  # 构建数据路径
        file_name = f"client{i+1}_preprocessed.csv"  # 文件名
        client_path = os.path.join(data_path, file_name)  # 完整文件路径
        client_train = pd.read_csv(client_path)  # 读取CSV文件
        client_train = client_train.to_numpy()  # 转换为numpy数组
        return client_train

    '''
    从KDD数据集中获取数据 (CSV文件)
    '''
    def get_data_kdd(self, i):
        directory = os.getcwd()
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        file_name = f"client{i+1}_kdd_std.csv"
        client_path = os.path.join(data_path, file_name)
        client_train = pd.read_csv(client_path)
        client_train = client_train.to_numpy()
        return client_train
    
    '''
    从NSL-KDD数据集中获取数据 (CSV文件)
    '''
    def get_data_snl_kdd(self):
        directory = os.getcwd()
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        file_name = f"nslkdd_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['dst_bytes'])  # 按照'dst_bytes'排序
        client_train = client_train.drop(['Unnamed: 0', 'outcome'], axis=1)  # 删除不必要的列
        print("Sorted!!!!!")
        return client_train

    '''
    从UNSW-NB15数据集中获取数据 (CSV文件)
    '''
    def get_data_unsw_nb15(self):
        directory = os.getcwd()
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        file_name = f"unswnb15_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['ct_srv_src'])  # 按照'ct_srv_src'排序
        client_train = client_train.drop(["Unnamed: 0"], axis=1)  # 删除不必要的列
        print("Created Non-iid Data!!!!!")
        return client_train

    '''
    从IoT23数据集中获取数据 (CSV文件)
    '''
    def get_data_Iot23(self):
        directory = os.getcwd()
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        file_name = f"iot23_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['duration'])  # 按照'duration'排序
        client_train = client_train.drop(["Unnamed: 0"], axis=1)  # 删除不必要的列
        print("Created Non-iid Data!!!!!")
        return client_train

    '''
    从ToN数据集中获取数据 (CSV文件)
    '''
    def get_data_Ton(self):
        directory = os.getcwd()
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        file_name = f"ton_train_normal_49.csv"
        client_path = os.path.join(data_path, file_name)
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['src_port'])  # 按照'src_port'排序
        client_train = client_train.drop(["Unnamed: 0"], axis=1)  # 删除不必要的列
        print("Created Non-iid Data!!!!!")
        return client_train

    '''
    数据预处理步骤
    '''
    def prep_data(self, dataX):
        change_dataX = dataX.copy()  # 复制数据
        featuresToScale = change_dataX.columns  # 获取所有特征列
        sX = StandardScaler(copy=True)  # 创建标准化缩放器
        change_dataX.loc[:, featuresToScale] = sX.fit_transform(change_dataX[featuresToScale])  # 对特征进行标准化
        return change_dataX

    '''
    将数据分配给客户端
    '''
    def get_client_data(self, data, factor, i):
        factor = int(factor)
        dataX = data[factor * i:factor * (i + 1)].copy()  # 获取对应客户端的数据片段
        client_data = self.prep_data(dataX)  # 对数据进行预处理
        client_data = client_data.to_numpy()  # 转换为numpy数组
        return client_data
    
    '''
    训练模型
    '''
    def train(self):
        current_loss = 0  # 当前损失值
        acc_score = 0  # 当前准确率
        losses_to_file = []  # 保存每次迭代的损失值
        acc_score_to_file = []  # 保存每次迭代的准确率
        acc_score_to_file.append(acc_score)  # 初始化准确率为0
        self.selected_users = self.select_users(1000, 1)  # 模式1，选择所有用户参与第一轮训练

        start_time = time.time()  # 开始计时
        for glob_iter in range(self.num_glob_iters):
            if self.experiment:
                self.experiment.set_epoch(glob_iter + 1)  # 设置实验轮次
            print("-------------Round number: ", glob_iter, " -------------")

            self.send_pca()  # 发送PCA信息给客户端

            # 每轮迭代评估模型
            current_loss = self.evaluate()  # 评估损失值
            current_loss = current_loss.item()
            losses_to_file.append(current_loss)

            # 随机选择一部分用户参与本轮训练
            self.selected_users = self.select_users(glob_iter, self.user_fraction)

            # 在每个选中的用户上训练模型
            for user in self.selected_users:
                user.train(self.local_epochs)
        
            self.aggregate_pca()  # 聚合PCA结果

            
            Z = self.commonPCAz.detach().numpy()
            # detach() 是 PyTorch 中的一个方法，通常用于从计算图中分离张量（tensor），从而阻止梯度的传递。

            if self.dataset == "Unsw":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = unsw_nb15_test(Z)
            elif self.dataset == "Iot23":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = iot23_test(Z)
            elif self.dataset == "Ton":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = ton_test(Z)
            else:
                acc_score = nsl_kdd_test(Z)

            acc_score_to_file.append(accuracy_score)

        end_time = time.time()  # 结束计时

        # 提取公共表示
        Z = self.commonPCAz.detach().numpy()
    
        # 提取损失值和准确率
        losses_to_file = np.array(losses_to_file)
        acc_score_to_file = np.array(acc_score_to_file)

        # 保存公共表示、损失值和准确率到文件
        if self.algorithm == "FedPG":
            space = "Grassman"
        elif self.algorithm == "FedPE":
            space = "Euclidean"
    
        directory = os.getcwd()
        if self.dataset == "Unsw":
            data_path = os.path.join(directory, "results/UNSW")
            acc_path = os.path.join(data_path, "UNSW_acc")
            losses_path = os.path.join(data_path, "UNSW_losses")
            metrics_path = os.path.join(data_path, "UNSW_metrics_exp")
            model_dir = os.path.join(data_path, "UNSW_model")
        elif self.dataset == "Iot23":
            data_path = os.path.join(directory, "results/IOT23")
            acc_path = os.path.join(data_path, "IOT23_acc")
            metrics_path = os.path.join(data_path, "IOT23_metrics_exp")
            losses_path = os.path.join(data_path, "IOT23_losses")
            model_dir = os.path.join(data_path, "IOT23_model")
        elif self.dataset == "Ton":
            data_path = os.path.join(directory, "results/TON")
            acc_path = os.path.join(data_path, "TON_acc")
            losses_path = os.path.join(data_path, "TON_losses")
            metrics_path = os.path.join(data_path, "TON_metrics_exp")
            model_dir = os.path.join(data_path, "TON_model")
        else:
            data_path = os.path.join(directory, "results/KDD")
            acc_path = os.path.join(data_path, "KDD_acc")
            losses_path = os.path.join(data_path, "KDD_losses")
            metrics_path = os.path.join(data_path, "KDD_metrics_exp")
            model_dir = os.path.join(data_path, "KDD_model")

        # 检查并创建目录
        if not os.path.exists(acc_path):
            os.makedirs(acc_path)
        if not os.path.exists(losses_path):
            os.makedirs(losses_path)
        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        acc_file_name = f'{space}_acc_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}_localEpochs_{self.local_epochs}'
        acc_file_path = os.path.join(acc_path, acc_file_name)

        
        losses_file_name = f"{space}_losses_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}_localEpochs_{self.local_epochs}"
        losses_file_path = os.path.join(losses_path, losses_file_name)
        model_name = f'{space}_model_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}_localEpochs_{self.local_epochs}'
        model_path = os.path.join(model_dir, model_name)

        # 保存准确率、损失值和模型到文件
        np.save(acc_file_path, acc_score_to_file)
        np.save(losses_file_path, losses_to_file)
        np.save(model_path, Z)
        print(f"------------Final Test results------------")
        training_time = end_time - start_time
        print(f"training time: {training_time} seconds")

        if self.dataset == "Unsw":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = unsw_nb15_test(Z)
        elif self.dataset == "Iot23":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = iot23_test(Z)        
        elif self.dataset == "Ton":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = ton_test(Z)
        else:
            nsl_kdd_test(Z)

        # 保存实验指标
        metrics_file_name = f"{self.dataset}_{self.algorithm}_{self.experiment_type}.csv"
        metrics_exp_file_path = os.path.join(metrics_path, metrics_file_name)
        data_row = [
            self.num_clients,
            self.num_glob_iters,
            self.local_epochs,
            self.dim,
            current_loss,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            fng,
            training_time
        ]
        metrics_exp_store(metrics_exp_file_path, data_row)
        print("Completed training!!!")
        print(f"------------------------------------------")