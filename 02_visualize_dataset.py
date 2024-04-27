import torch
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath)

def plot_trajectories(data, n):
    """Plot the first n trajectories from the dataset."""
    plt.figure(figsize=(10, 8))
    for i in range(min(n, len(data['yts']))):  # Ensure we do not exceed available trajectories
        traj = data['yts'][i]
        plt.scatter(traj[:, 0], traj[:, 1], label=f'Trajectory {i+1}', s=10)  # s is the size of points

    plt.title(f'First {n} Trajectories out of {len(data["yts"])} Trajectories')
    plt.xlabel('Theta (rad)')
    plt.ylabel('p_theta')
    plt.legend()
    plt.grid(True)
    plt.savefig('pendulum_train_trajectories.png')

if __name__ == '__main__':
    filepath = 'pendulum_train_trajectories.pth'  # Path to the data file
    n = 5  # Number of trajectories to plot

    data = load_data(filepath)
    plot_trajectories(data, n)
