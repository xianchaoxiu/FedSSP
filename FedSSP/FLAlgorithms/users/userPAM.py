import torch
import copy
import numpy as np
from FLAlgorithms.users.userbase import User

class UserPAM():
    def __init__(self, algorithm, device, id, train_data, commonPCA, learning_rate, beta, tau, lambda1, lambda2, local_epochs, dim,p,q):
        # Initialize local variables
        self.localPCA = copy.deepcopy(commonPCA)
        self.localW = copy.deepcopy(commonPCA)
        self.localU = copy.deepcopy(commonPCA)
        self.localV = copy.deepcopy(commonPCA)
        self.localZ = copy.deepcopy(commonPCA)
        
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
        self.p = p
        self.q = q
        
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
        self.localW.requires_grad_(True)

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
        


    def update_W(self):
        """Update W using Riemannian gradient descent on Grassmann manifold"""
        self.localW.requires_grad_(True)
        
        S = torch.matmul(self.train_data, self.train_data.T)
        gradient = -2 * torch.matmul(S, self.localW) + \
                  self.beta1 * (self.localW - self.localU) + \
                  self.beta2 * (self.localW - self.localV) + \
                  self.beta3 * (self.localW - self.localZ) 

        # Project gradient onto tangent space of Grassmann manifold
        
        projection = gradient - torch.matmul(
            self.localW,
            (torch.matmul(self.localW.T, gradient) + 
             torch.matmul(gradient.T, self.localW)) / 2
        )
        
        # Update W using Riemannian gradient descent
        temp = self.localW - self.learning_rate * projection
        
        # Retract back to manifold using QR decomposition
        q, r = torch.linalg.qr(temp)
        self.localW = q.data.clone()
        
        # self.localPCA = self.localW
        # print("W:", np.linalg.norm(self.localW, ord=2))


    def update_U(self):
        """Update U using proximal operator"""
        # user_input = input("请输入内容后按回车继续：")
        # print("U:", np.linalg.norm(self.localU, ord=2))
        A = (self.beta1 * self.localW + self.tau2 * self.localU) / (self.beta1 + self.tau2)
        threshold = self.lambda1 / (self.beta1 + self.tau2)
        for i in range(A.shape[0]):  # 逐行计算
            self.localU[i] = self.proxmlq(A[i], threshold, self.p)  #传入参数

        # self.localPCA = self.localU
        # print("U:", np.linalg.norm(self.localU, ord=2))

    def update_V(self):
        """Update V using row-wise proximal operator"""
        B = (self.beta2 * self.localW + self.tau3 * self.localV) / (self.beta2 + self.tau3)
        threshold = self.lambda2 / (self.beta2 + self.tau3)
        
        # Apply proximal operator row-wise
        for i in range(B.shape[0]):
            row_norm = torch.norm(B[i], p=2)
            if row_norm == 0:
                self.localV[i] = 0
            else:
                self.localV[i] = self.proxmlq(row_norm,threshold, self.q) * B[i] / row_norm  #传入参数
        
        self.localPCA = self.localV                           # settings: 使用行稀疏W作为参数，传递到全局
        # print("V:", np.linalg.norm(self.localV, ord=2))

    def set_commonPCA(self, commonPCA):
        """Update Z (global consensus variable)"""
        self.localZ = commonPCA.data.clone()

    def train_error_and_loss(self):
        """Calculate training error and loss"""
        residual = torch.matmul(
            torch.eye(self.localW.shape[0]) - torch.matmul(self.localW, self.localW.T),
            self.train_data
        )
        reconstruction_loss = torch.norm(residual, p="fro") ** 2
        if self.p == 0.0:
            loss_U = torch.nonzero(self.localU).size(0)
        else:
            loss_U = torch.sum(torch.pow(self.localU, self.p)) 
            
        if self.q == 0.0:
            loss_V = torch.sum(torch.any(self.localV != 0, dim=1)).item()
        else:
            loss_V = torch.sum(torch.norm(self.localV, p=self.q, dim=1))

        loss_train = (reconstruction_loss + self.lambda1 * loss_U + self.lambda2 * loss_V) / self.train_samples
        return loss_train, self.train_samples
    def train(self, epochs):
        """Train using PAM framework"""
        for _ in range(self.local_epochs):
            # Update variables according to PAM framework
            self.update_W()
            self.update_U()
            self.update_V()
            
            # Calculate loss for monitoring
            #residual = torch.matmul(
            #     torch.eye(self.localW.shape[0]) - torch.matmul(self.localW, self.localW.T),
            #     self.train_data
            # )
            # reconstruction_loss = torch.norm(residual, p="fro") ** 2 / self.train_samples
            

            # loss_U = torch.nonzero(self.localU).size(0)
            # loss_V = torch.sum(torch.any(self.localV != 0, dim=1)).item()

            self.loss ,_ = self.train_error_and_loss()
            