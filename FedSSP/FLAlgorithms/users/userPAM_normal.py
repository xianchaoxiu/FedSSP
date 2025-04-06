import torch
import copy
import numpy as np
from FLAlgorithms.users.userbase import User

class UserPAM():
    def __init__(self, algorithm, device, id, train_data, commonPCA, learning_rate, beta, tau, lambda1, lambda2, local_epochs, dim):
        # Initialize local variables
        self.localPCA = copy.deepcopy(commonPCA)
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

    def update_W(self):
        """Update W using Riemannian gradient descent on Grassmann manifold"""
        self.localPCA.requires_grad_(True)
        
        # Calculate residual loss
        # residual = torch.matmul(torch.eye(self.localPCA.shape[0]) - 
        #                       torch.matmul(self.localPCA, self.localPCA.T), 
        #                       self.train_data)
        #----------------------------------
        # print("程序运行中...")
        # exit() 
        #Calculate gradient as per equation (gradient_W)
        S = torch.matmul(self.train_data, self.train_data.T)
        gradient = -2 * torch.matmul(S, self.localPCA) + \
                  self.beta1 * (self.localPCA - self.localU) + \
                  self.beta2 * (self.localPCA - self.localV) + \
                  self.beta3 * (self.localPCA - self.localZ) + \
                  self.tau1 * (self.localPCA - self.localPCA)

        # Project gradient onto tangent space of Grassmann manifold
        
        projection = gradient - torch.matmul(
            self.localPCA,
            (torch.matmul(self.localPCA.T, gradient) + 
             torch.matmul(gradient.T, self.localPCA)) / 2
        )
        
        # Update W using Riemannian gradient descent
        temp = self.localPCA - self.learning_rate * projection
        
        # Retract back to manifold using QR decomposition
        q, r = torch.linalg.qr(temp)
        self.localPCA = q.data.clone()


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