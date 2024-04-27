import numpy as np
import torch
from torch import nn
from math import floor, sqrt

class SSGP(torch.nn.Module):
    def __init__(self, input_dim, basis, friction, K):
        super(SSGP, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([1e-1]))
        self.a = nn.Parameter(torch.ones(input_dim) * 1e-1)
        self.b = nn.Parameter(1e-4 * (torch.rand(basis * 2) - 0.5))
        self.SqrtC = nn.Parameter(torch.ones(basis,basis)*1e-2+torch.eye(basis)*1e-2)
        self.sigma_0 = nn.Parameter(torch.tensor([1e-0]))
        self.lam = nn.Parameter(torch.ones(input_dim) * 1.5)
        self.eta = nn.Parameter(torch.tensor([1e-16])) if friction else torch.tensor([0.0])
        self.S = self.permutation_tensor(input_dim)
        tmp = torch.normal(0, 1, size=(basis // 2, input_dim))
        self.epsilon = torch.vstack([tmp, -tmp])
        self.d = input_dim
        self.num_basis = basis
        self.K = K

    # def sampling_epsilon_f(self):
    #     sqrt_C = torch.block_diag(self.SqrtC,self.SqrtC)
    #     epsilon = torch.tensor(np.random.normal(0, 1, size=(1,sqrt_C.shape[0]))).T
    #     self.w = self.b + (sqrt_C @ epsilon).squeeze()
    #     for i in range(self.K):
    #         epsilon = torch.tensor(np.random.normal(0, 1, size=(1,sqrt_C.shape[0]))).T
    #         self.w += self.b + (sqrt_C @ epsilon).squeeze()
    #     self.w = self.w/self.K
    def sampling_epsilon_f(self):
        sqrt_C = torch.block_diag(self.SqrtC, self.SqrtC)
        # Ensure the data type matches other model parameters, assuming they are float32
        epsilon = torch.tensor(np.random.normal(0, 1, size=(1, sqrt_C.shape[0])), dtype=torch.float32).T
        self.w = self.b + (sqrt_C @ epsilon).squeeze()
        for i in range(self.K):
            epsilon = torch.tensor(np.random.normal(0, 1, size=(1, sqrt_C.shape[0])), dtype=torch.float32).T
            self.w += self.b + (sqrt_C @ epsilon).squeeze()
        self.w = self.w / self.K

    def mean_w(self):
        self.w = self.b * 1

    def neg_loglike(self, batch_x, pred_x):
        n_samples, n_points, dammy, input_dim = batch_x.shape
        likelihood = ( (-(pred_x-batch_x)**2/self.sigma**2/2).nansum()
                    - torch.log(self.sigma**2)/2*n_samples*n_points*input_dim)
        return -likelihood

    def KL_x0(self, x0):
        n, d = x0.shape
        S = torch.diag(self.a**2)
        return 0.5 * ((x0 * x0).sum() / n + d * torch.trace(S) - d * torch.logdet(S) - d)

    def KL_w(self):
        num = self.b.shape[0]
        C = self.SqrtC @ self.SqrtC.T
        C = torch.block_diag(C,C)
        term3 = (self.b*self.b).sum() / (self.sigma_0**2 / num * 2)
        term2 = torch.diag(C).sum() / (self.sigma_0**2 / num * 2)
        term1_1 = torch.log(self.sigma_0**2 / num * 2) * num
        term1_2 = torch.logdet(C)
        return .5*( term1_1 - term1_2 + term2 + term3)

    def sampling_x0(self, x0):
        n, dammy, d = x0.shape
        return (x0 + torch.sqrt(torch.stack([self.a**2]*n).reshape([n,1,d]))
                * (torch.normal(0,1, size=(x0.shape[0],1,x0.shape[2]))))

    def permutation_tensor(self, n):
        S = torch.eye(n)
        return torch.cat([S[n // 2:], -S[:n // 2]], dim=0)

    def forward(self, t, x):
        # Compute the sqrt(Var) of the kernel's spectral density
        s = self.epsilon @ torch.diag((1 / torch.sqrt(4*torch.pi**2 * self.lam**2)))

        # Dissipation matrix
        R = torch.eye(self.d)
        R[:int(self.d/2),:int(self.d/2)] = 0

        # Compute te Symplectic Random Fourier Feature (SRFF) matrix (Psi)
        Psi = 2*torch.pi*((self.S-self.eta**2*R)@s.T).T
        x = x.squeeze()
        samples = x.shape[0]
        sim = 2*torch.pi*s@x.squeeze().T
        basis_s = -torch.sin(sim); basis_c = torch.cos(sim)

        # deterministic
        tmp = []
        for i in range(self.d):
            tmp.extend([Psi[:,i]]*samples)
        tmp = torch.stack(tmp).T
        aug_mat = torch.vstack([tmp,tmp])
        aug_s = torch.hstack([basis_s]*self.d)
        aug_c = torch.hstack([basis_c]*self.d)
        aug_basis = torch.vstack([aug_s, aug_c])
        PHI = aug_mat * aug_basis
        aug_W = torch.stack([self.w]*samples*self.d).T
        F = PHI * aug_W
        f = torch.vstack(torch.split(F.sum(axis=0),samples)).T
        return f.reshape([samples,1,self.d])

    def sample_hamiltonian(self, x):
        print('x has shape:', x.shape)
        # Compute the sqrt(Var) of the kernel's spectral density
        s = self.epsilon @ torch.diag((1 / torch.sqrt(4*torch.pi**2 * self.lam**2)))
        print('s has shape:', s.shape)

        # Compute te Symplectic Random Fourier Feature (SRFF) matrix (Psi)
        sim = 2*torch.pi*s@x.squeeze().T
        print('sim has shape:', sim.shape)
        basis_c = torch.cos(sim); basis_s = torch.sin(sim)
        print('basis_c has shape:', basis_c.shape)
        print('basis_s has shape:', basis_s.shape)

        samples = x.shape[0]
        print('samples:', samples)

        aug_basis = torch.vstack([basis_c, basis_s])
        print('aug_basis has shape:', aug_basis.shape)

        aug_W = torch.stack([self.w]*samples).T
        print('aug_W has shape:', aug_W.shape)

        H_aug = aug_basis * aug_W
        print('H has shape:', H_aug.shape)

        H = torch.vstack(torch.split(H_aug.sum(axis=0),samples)).T
        print('H has shape:', H.shape)

        return H