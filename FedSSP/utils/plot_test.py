# 导入 plot_utils 模块
from plot_utils import plot_summary_one_figure

# 设置实验参数
num_users = 100
loc_ep1 = 5
Numb_Glob_Iters = 100
lamb = [0.1, 0.01]
learning_rate = [0.01, 0.001]
beta = [0.1, 0.01]
algorithms_list = ["FedAvg", "pFedMe"]
batch_size = [32, 64]
dataset = "MNIST"
k = [0, 0]  # 如果算法不需要 k 参数，可以传入相同的值
personal_learning_rate = [0.01, 0.001]  # 如果算法不需要个人学习率，可以传入相同的值

# 调用 plot_summary_one_figure 函数
plot_summary_one_figure(
    num_users=num_users,
    loc_ep1=loc_ep1,
    Numb_Glob_Iters=Numb_Glob_Iters,
    lamb=lamb,
    learning_rate=learning_rate,
    beta=beta,
    algorithms_list=algorithms_list,
    batch_size=batch_size,
    dataset=dataset,
    k=k,
    personal_learning_rate=personal_learning_rate
)