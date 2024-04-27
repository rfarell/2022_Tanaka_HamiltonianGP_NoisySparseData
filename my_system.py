import torch
from torch import nn
from torchdiffeq import odeint

class ODEPendulum(nn.Module):
    def __init__(self, mass, gravity, length, friction):
        super().__init__()
        # Pendulum properties
        self.m = mass
        self.g = gravity
        self.l = length
        self.r = friction  # Damping factor
        # Skew-symmetric matrix for Hamiltonian mechanics
        self.S = torch.tensor([[0, 1], [-1, 0]]).float()

    def hamiltonian(self, coords):
        # Kinetic + Potential Energy
        q, p = coords[:, 0], coords[:, 1]
        H = p.pow(2) / (2 * self.m * self.l ** 2) + self.m * self.g * self.l * (1 - torch.cos(q))
        return H
    
    def forward(self, t, x):
        # Calculate the Hamiltonian
        H = self.hamiltonian(x)
        # Compute gradient of Hamiltonian
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        # Hamiltonian vector field calculation
        field = dH @ self.S.t()
        # Zero out the derivative with respect to theta to prevent damping effect on it
        dH[:, 0] = 0
        # Apply damping only to the momentum component
        field -= self.r * dH
        return field

class PendulumDataset:
    def __init__(self, mass, gravity, length, friction, T, timescale, samples, q_lower, q_upper, sigma):
        """
        Initializes parameters for simulating a pendulum dataset.
        
        Parameters:
            mass (float): Mass of the pendulum bob.
            gravity (float): Acceleration due to gravity.
            length (float): Length of the pendulum.
            friction (float): Damping coefficient.
            T (float): Total time of simulation.
            timescale (int): Number of time steps per unit time.
            samples (int): Number of initial conditions to simulate.
            q_lower (float): Lower bound of the pendulum's initial angle.
            q_upper (float): Upper bound of the pendulum's initial angle.
            sigma (float): Standard deviation of measurement noise.
        """
        self.ode = ODEPendulum(mass, gravity, length, friction)
        self.samples = samples
        self.T = T
        self.timescale = timescale
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.sigma = sigma

    # def get_initial_conditions(self):
    #     """
    #     Generates random initial conditions for the pendulum's angle, with zero initial momentum.
        
    #     Returns:
    #         torch.Tensor: Initial conditions [theta, p_theta].
    #     """
    #     q0s = torch.linspace(self.q_lower, self.q_upper, self.samples).unsqueeze(1)
    #     p0s = torch.zeros(self.samples, 1)
    #     return torch.cat([q0s, p0s], dim=1).requires_grad_(True)

    def get_initial_conditions(self):
        """
        Generates random initial conditions for the pendulum's angle, with zero initial momentum.
        
        Returns:
            torch.Tensor: Initial conditions [theta, p_theta].
        """
        # Uniformly random angles between q_lower and q_upper
        q0s = (self.q_upper - self.q_lower) * torch.rand(self.samples, 1) + self.q_lower
        # Zero initial momenta
        p0s = torch.zeros(self.samples, 1)
        # Concatenate angles and momenta to form initial conditions
        initial_conditions = torch.cat([q0s, p0s], dim=1)
        initial_conditions.requires_grad_(True)  # Enable gradient computation for initial conditions
        return initial_conditions

    def generate_trajectories(self):
        """
        Simulates the pendulum dynamics and generates trajectories with added noise.
        
        Returns:
            dict: Dictionary containing the simulated state variables and their derivatives.
        """
        x0s = self.get_initial_conditions()
        t = torch.linspace(0, self.T, int(self.T * self.timescale + 1))
        xts = odeint(self.ode, x0s, t, method='dopri5', atol=1e-8, rtol=1e-8)
        
        # Add Gaussian noise to each state variable to model measurement errors
        noise = torch.randn(xts.shape) * self.sigma
        yts = xts + noise
        yts = yts.permute(1, 0, 2)
        return {'yts': yts.detach(), 't': t.detach()}
