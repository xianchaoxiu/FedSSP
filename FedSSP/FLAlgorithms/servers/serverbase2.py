import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import copy

class Server2:
    def __init__(self, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, times):
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.L_k = ro
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc = [], [], []
        self.times = times
        self.dim = dim

    def send_pca(self):
        assert (self.users is not None and len(self.users) > 0)
        # print("check Z", torch.matmul(self.commonPCAz.T,self.commonPCAz))
        # for user in self.users:
        for user in self.selected_users:
            # print("user_id", user.id)
            user.set_commonPCA(self.commonPCAz)
    
    def add_pca(self, user, ratio):
        # ADMM update
        # self.commonPCAz += ratio*(user.localPCA + 1/user.ro * user.localY)
        # simplified ADMM update
        # print("simplified ADMM update")
        self.commonPCAz += ratio*(user.localPCA) 

    def aggregate_pca(self):  
        # 确保用户列表已初始化且不为空，这是执行PCA聚合的前提条件
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        # 开始计算所有选定用户训练样本的总数
        for user in self.selected_users:
            total_train += user.train_samples
            # print("user_id", user.id)  # 调试时打印用户ID
           # 初始化共同的PCA变量为零，以便开始聚合过程
        self.commonPCAz = torch.zeros(self.commonPCAz.shape)
        # 根据每个用户的训练样本数量，按比例聚合PCA结果
        for user in self.selected_users:
            self.add_pca(user, user.train_samples / total_train)

    def aggregate_pca_doubleSparse(self,beta3,tau4):  
        """ 进行 PCA 结果聚合，考虑用户的训练样本数、额外权重、beta3 和 Zk """
        assert (self.users is not None and len(self.users) > 0)
        print("check Z:", np.linalg.norm(self.commonPCAz, ord=2))

        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        
        # 计算加权和分母
        total_weighted_beta = sum(user.train_samples / total_train * beta3 for user in self.selected_users) + tau4
        
        # 初始化聚合变量
        commonPCAz_old = copy.deepcopy(self.commonPCAz)
        self.commonPCAz = torch.zeros_like(self.commonPCAz)  

        # 计算分子
        for user in self.selected_users:
            ratio = (user.train_samples / total_train * beta3) / total_weighted_beta
            self.add_pca(user, ratio)

        # 加上 Zk 贡献
        self.commonPCAz += (tau4 / total_weighted_beta) * commonPCAz_old
    
    def select_users(self, round, fac_users):
        if(fac_users == 1):
            print("Distribute global model to all users")
            # for user in self.users:
            #     print("user_id", user.id)
            return self.users
        num_users = int(fac_users * len(self.users))
        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # Save loss, accurancy to h5 fiel
    def train_error_and_loss(self):
        num_samples = []
        losses = []
        for c in self.selected_users:
            cl, ns = c.train_error_and_loss() 
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]

        return ids, num_samples, losses

    def evaluate(self):
        stats_train = self.train_error_and_loss()
        # print(f"stats_train: {stats_train}")
        train_loss = sum(stats_train[2])/len(self.users)
        self.rs_train_loss.append(train_loss)
        if(self.experiment):
            self.experiment.log_metric("train_loss",train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Trainning Loss: ",train_loss)
        return train_loss
    
    def save_results(self):
        dir_path = "./results"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        alg = self.dataset[1] + "ADMM" + "_" + str(self.learning_rate)  + "_" + str(self.L_k) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs) 
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()

