from my_system import PendulumDataset
import torch

if __name__ == '__main__':
    mass = 1.0  # kg
    gravity = 9.81  # m/s^2
    length = 1.0  # m
    friction = 0.01  # kg/s
    T = 10.0  # s
    timescale = 10  # observations/second
    train_samples = 100
    test_samples = 100
    q_lower = 0.1  # rad
    q_upper = 1.1  # rad
    sigma = 0.01  # m

    # Create training dataset
    train_dataset = PendulumDataset(mass, gravity, length, friction, T, timescale, train_samples, q_lower, q_upper, sigma)
    train_data = train_dataset.generate_trajectories()
    torch.save(train_data, 'pendulum_train_trajectories.pth')

    # Create testing dataset
    test_dataset = PendulumDataset(mass, gravity, length, friction, T, timescale, test_samples, q_lower, q_upper, sigma)
    test_data = test_dataset.generate_trajectories()
    torch.save(test_data, 'pendulum_test_trajectories.pth')

    # Optionally, load the data again to check
    loaded_train_data = torch.load('pendulum_train_trajectories.pth')
    loaded_test_data = torch.load('pendulum_test_trajectories.pth')
    print(f'Train data keys: {loaded_train_data.keys()}')
    print(f'Test data keys: {loaded_test_data.keys()}')