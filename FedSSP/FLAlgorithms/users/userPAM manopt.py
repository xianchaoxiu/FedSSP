import torch
import copy
import numpy as np
from FLAlgorithms.users.userbase import User
import pymanopt
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import TrustRegions

class UserPAM():
    def __init__(self, algorithm, device, id, train_data, commonPCA, learning_rate, beta, tau, lambda1, lambda2, local_epochs, dim):
        # Initialize local variables
        self.localPCA = copy.deepcopy(commonPCA).float()
        self.localU = copy.deepcopy(commonPCA).float()
        self.localV = copy.deepcopy(commonPCA).float()
        self.localZ = copy.deepcopy(commonPCA).float()
        
        # PAM parameters
        self.beta1 = beta
        self.beta2 = beta
        self.beta3 = beta
        self.tau1 = tau
        self.tau2 = tau
        self.tau3 = tau
        self.tau4 = tau
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Other parameters
        self.device = device
        self.id = id
        self.train_samples = len(train_data)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.dim = dim
        self.train_data = train_data.T
        self.algorithm = algorithm
        
        # Enable gradient computation
        self.localPCA.requires_grad_(True)

    def prox_l1(self, x, threshold):
        """Proximal operator for l1 norm (element-wise soft thresholding)"""
        return torch.sign(x) * torch.max(torch.abs(x) - threshold, torch.zeros_like(x))

    def proxmlq(self, a, alam, q):

        px = torch.zeros_like(a)  # 改为PyTorch的zeros_like
        
        if q == 0:
            t = torch.sqrt(torch.tensor(2 * alam, device=a.device))
            T = torch.abs(a) > t
            px[T] = a[T]
        
        elif q == 0.5:
            t = (3/2) * alam**(2/3)
            T = torch.abs(a) > t
            aT = a[T]
            phi = torch.acos((alam/4) * (3.0 / torch.abs(aT))**(3/2))
            px[T] = (4/3) * aT * (torch.cos((torch.pi - phi)/3))**2
        
        elif q == 2/3:
            t = (128/27)**(1/4) * (alam)**(3/4)
            T = torch.abs(a) > t
            aT = a[T]
            tmp1 = aT**2 / 2
            tmp2 = torch.sqrt(tmp1**2 - (8 * alam/9)**3)
            phi = torch.pow(tmp1 + tmp2, 1/3) + torch.pow(tmp1 - tmp2, 1/3)
            px[T] = torch.sign(aT) / 8 * (torch.sqrt(phi) + torch.sqrt(2 * torch.abs(aT) / torch.sqrt(phi) - phi))**3
        
        else:
            print('Input \'q\' is incorrect!!! \n')
            print('Please re-enter \'q\' one value of {0, 0.5, 2/3}!!!\n')
        
        return px
        

    # def prox_l2(self, x, threshold):
    #     """Proximal operator for l2 norm (vector soft thresholding)"""
    #     norm = torch.norm(x, p=2)
    #     if norm > threshold:
    #         return x * (1 - threshold/norm)
    #     return torch.zeros_like(x)

    # def costW(self, Wi, Xi, Ui, Vi, Z, beta1, beta2, beta3, device):
    #     # Convert all inputs to float64 dtype
    #     Wi = Wi.to(dtype=torch.float64, device=device)
    #     Xi = Xi.to(dtype=torch.float64, device=device)
    #     Ui = Ui.to(dtype=torch.float64, device=device)
    #     Vi = Vi.to(dtype=torch.float64, device=device)
    #     Z = Z.to(dtype=torch.float64, device=device)
        
    #     # Create identity matrix
    #     I = torch.eye(Wi.shape[0], device=device, dtype=torch.float64)
        
    #     # Calculate cost terms
    #     projection = torch.matmul(I - torch.matmul(Wi, Wi.T), Xi)
    #     term1 = torch.norm(projection, 'fro').pow(2)
    #     term2 = (beta1 / 2) * torch.norm(Wi - Ui, 'fro').pow(2)
    #     term3 = (beta2 / 2) * torch.norm(Wi - Vi, 'fro').pow(2)
    #     term4 = (beta3 / 2) * torch.norm(Wi - Z, 'fro').pow(2)
        
    #     return term1 + term2 + term3 + term4
    
    def update_W(self):
        """Manopt trust region algorithm for W (重构版)"""
        # 创建流形
        manifold = Grassmann(self.localPCA.shape[0], self.localPCA.shape[1])
        
        # 准备输入数据（转换为PyTorch张量）
        X = self.train_data.clone().detach().to(self.device)
        U = self.localU.clone().detach().to(self.device)
        V = self.localV.clone().detach().to(self.device)
        Z = self.localZ.clone().detach().to(self.device)

        # 使用PyTorch后端定义cost函数
        @pymanopt.function.pytorch(manifold)
        def cost(w):
            # 确保输入是矩阵形式
            assert not torch.is_complex(w), "出现复数类型！"
            w_torch = w.view(self.localPCA.shape).float()
            
            # 重构cost计算（保持原有计算逻辑）
            projection = torch.matmul(
                torch.eye(w_torch.shape[0], device=self.device) - torch.matmul(w_torch, w_torch.T),
                X
            )
            term1 = torch.norm(projection, 'fro').pow(2)
            term2 = (self.beta1 / 2) * torch.norm(w_torch - U, 'fro').pow(2)
            term3 = (self.beta2 / 2) * torch.norm(w_torch - V, 'fro').pow(2)
            term4 = (self.beta3 / 2) * torch.norm(w_torch - Z, 'fro').pow(2)
            return term1 + term2 + term3 + term4
        @pymanopt.function.pytorch(manifold)    
        def grad(w):
            
            w_torch = w.view(self.localPCA.shape).float()

            S = torch.matmul(self.train_data, self.train_data.T)
            gradient = -2 * torch.matmul(S, w_torch) + \
               self.beta1 * (w_torch - self.localU) + \
               self.beta2 * (w_torch - self.localV) + \
               self.beta3 * (w_torch - self.localZ) + \
               self.tau1 * (w_torch - w_torch)
            return gradient

        # 创建优化问题（自动计算梯度）
        problem = pymanopt.Problem(
            manifold,
            cost,
            #euclidean_gradient=grad  # 自动计算梯度
        )

        # 配置优化器
        optimizer = TrustRegions(
            verbosity=2,
            max_iterations=100,
            min_step_size=1e-6
           
        )
        
        # 执行优化
        result = optimizer.run(problem)
        
        # 处理优化结果
        w_opt = torch.tensor(
            result.point,
            # dtype=torch.float64,
            device=self.device
        ).view_as(self.localPCA)
        
        return w_opt

    def update_U(self):
        """Update U using proximal operator"""
        A = (self.beta1 * self.localPCA + self.tau2 * self.localU) / (self.beta1 + self.tau2)
        threshold = self.lambda1 / (self.beta1 + self.tau2)
        for i in range(A.shape[0]):  # 逐行计算
            self.localU[i] = self.proxmlq(A[i], threshold, 2/3)

    def update_V(self):
        """Update V using row-wise proximal operator"""
        B = (self.beta2 * self.localPCA + self.tau3 * self.localV) / (self.beta2 + self.tau3)
        threshold = self.lambda2 / (self.beta2 + self.tau3)
        
        # Apply proximal operator row-wise
        for i in range(B.shape[0]):
            row_norm = torch.norm(B[i], p=2)
            if row_norm == 0:
                self.localV[i] = 0
            else:
                self.localV[i] = self.proxmlq(row_norm,threshold, 2/3) * B[i] / row_norm

    def set_commonPCA(self, commonPCA):
        """Update Z (global consensus variable)"""
        self.localZ = commonPCA.data.clone()

    def train(self, epochs):
        """Train using PAM framework"""
        for _ in range(self.local_epochs):
            # Update variables according to PAM framework
            self.update_W()
            self.update_U()
            self.update_V()
            
            # Calculate loss for monitoring
            residual = torch.matmul(
                torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T),
                self.train_data
            )
            self.loss = torch.norm(residual, p="fro") ** 2 / self.train_samples
            
            # Add regularization terms
            self.loss += (self.lambda1 * torch.norm(self.localU, p=1) + 
                         self.lambda2 * torch.sum(torch.norm(self.localV, p=2, dim=1)))

    def train_error_and_loss(self):
        """Calculate training error and loss"""
        residual = torch.matmul(
            torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T),
            self.train_data
        )
        loss_train = torch.norm(residual, p="fro") ** 2 / self.train_samples
        return loss_train, self.train_samples