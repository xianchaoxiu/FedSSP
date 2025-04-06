"""
This code is based on the FedPCA_AnomalyDetection project by dual-grp.
Original repository: https://github.com/dual-grp/FedPCA_AnomalyDetection
Authors: Tung-Anh Nguyen, Long Tan Le, Tuan Dung Nguyen, Wei Bao, Suranga Seneviratne, Choong Seon Hong, Nguyen H. Tran
Paper: Fed PCA: Federated PCA on Grassmann Manifold for Anomaly Detection in IoT Networks [IEEE Transactions on Networking]
Link to paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10593810
"""

#!/usr/bin/env python
#from comet_ml import Experiment
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.myAbnormalDetection import AbnormalDetection
from utils.model_utils import read_data
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
from utils.options import args_parser

                                                                                 
# Create an experiment with your api key:
def main(experiment, dataset, algorithm, batch_size, learning_rate, ro, beta, tau, lambda1,lambda2, num_glob_iters,
         local_epochs, clients, numusers,dim, threshold, times, gpu, exp_type,p,q,fileName):
    
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    data = dataset

    server = AbnormalDetection(algorithm, experiment, device, data, learning_rate, ro, beta, tau, lambda1,lambda2, num_glob_iters, local_epochs, clients, numusers, dim, times, exp_type,threshold,p,q,fileName)
    
    server.train()

if __name__ == "__main__":
    args = args_parser()  # args_parser() 是 一个函数，用于解析命令行参数的
    print("=" * 80)
    print("Summary of training process:")   # 打印训练参数
    print("Algorithm: {}".format(args.algorithm)) # 打印算法名称
    print("Batch size: {}".format(args.batch_size)) # 打印批量大小
    print("Learing rate       : {}".format(args.learning_rate)) # 打印学习率
    print("Average Moving       : {}".format(args.ro)) # 打印平均移动
    print("Subset of users      : {}".format(args.subusers)) # 打印子用户
    print("Number of global rounds       : {}".format(args.num_global_iters)) # 打印全局轮数
    print("Number of local rounds       : {}".format(args.local_epochs)) # 打印本地轮数
    print("Dataset       : {}".format(args.dataset))
    print("=" * 80)

    experiment = 0
    
    main(
        experiment= experiment,
        dataset=args.dataset,
        algorithm = args.algorithm,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ro = args.ro,   
        beta = args.beta,
        tau = args.tau,
        lambda1 = args.lambda1,
        lambda2 = args.lambda2,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        clients = args.clients,
        numusers = args.subusers,
        dim = args.dim,
        threshold = args.threshold,
        times = args.times,
        gpu=args.gpu,
        exp_type=args.exp_type,
        p = args.p,   
        q = args.q,
        fileName = args.fileName
        )








