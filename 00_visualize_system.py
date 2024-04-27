import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from my_system import ODEPendulum
import torch
from torchdiffeq import odeint

def plot_pendulum_motion():
    """
    Animates the pendulum motion using the positions from the trajectory data alongside a phase-space plot.

    Parameters:
    - xts (torch.Tensor): Tensor containing the trajectory data with shape [num_steps, 1, 2]
                            where each entry is [theta, p_theta].
    - t (torch.Tensor): Tensor containing the time steps.
    - fps (int): Frames per second for the animation (default 100).
    """
    # Parameters for the pendulum
    mass = 1.0 # kg
    gravity = 9.81 # m/s^2
    length = 1.0 # m
    friction = 0.01 # kg/s

    ode = ODEPendulum(mass, gravity, length, friction)

    # Visualize the pendulum motion
    qp0 = torch.tensor([[0.5, 0.0]], requires_grad=True)
    t = torch.linspace(0, 10, 101)
    xts = odeint(ode, qp0, t, method='dopri5', atol=1e-8, rtol=1e-8)
    theta = xts[:, 0, 0].detach().numpy()
    p_theta = xts[:, 0, 1].detach().numpy()
    times = t.numpy()

    # Initialize the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('Pendulum Motion')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    line, = ax1.plot([], [], 'o-', lw=2, markersize=8)  # Line object to update in the animation for pendulum

    # Set up the phase space plot
    ax2.set_title('Phase Space')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-np.max(np.abs(p_theta))*1.1, np.max(np.abs(p_theta))*1.1)
    ax2.set_xlabel('$q$')
    ax2.set_ylabel('$p$')
    line_phase, = ax2.plot([], [], 'r-', alpha=0.5)  # Line object for phase space trajectory

    def init():
        """Initialize the background of both plots."""
        line.set_data([], [])
        line_phase.set_data([], [])
        return line, line_phase

    def update(frame):
        """Updates the position of the pendulum and phase space for each frame."""
        x = np.sin(theta[frame])
        y = -np.cos(theta[frame])
        line.set_data([0, x], [0, y])

        line_phase.set_data(theta[:frame + 1], p_theta[:frame + 1])
        return line, line_phase

    ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True, interval=50, repeat=False)

    # Save to file
    ani.save('./pendulum_animation.mp4', writer='ffmpeg', dpi=300)

    # Close the plot
    plt.close(fig)

if __name__ == '__main__':
    plot_pendulum_motion()