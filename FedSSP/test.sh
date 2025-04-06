    # dataset是 数据集名称
    # algorithm是 算法名称
    # batch_size是 批量大小
    # learning_rate是 学习率
    # ro是 平均移动
    # beta是 惩罚参数
    # tau是 PAM参数
    # lambda1是 正则化系数
    # lambda2是 正则化系数
    # num_glob_iters是 全局轮数
    # local_epochs是 本地轮数
    # clients是 客户端数量
    # numusers是 子用户数量
    # dim是 特征维度
    # threshold是 阈值
    # times是 运行次数
    # gpu是 GPU编号
    # exp_type是 运行类型

# random params to test

python3 main.py --algorithm FedPGSS --dataset Ton --learning_rate 0.001 --beta 100 --tau 0.01 --lambda1 0.001 --lambda2 0.001 --num_global_iters 3 --dim 5 --clients 20 --subusers 0.1 --local_epochs 30 --fileName "test" --exp_type Global_iter