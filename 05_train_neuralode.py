import torch
from collections import defaultdict
from torchdiffeq import odeint
import time
import copy
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

def load_checkpoint(filepath):
    """
    Load the model checkpoint from the given file path.
    """
    checkpoint = torch.load(filepath, map_location='cpu')  # Ensure it loads on CPU
    return checkpoint

def plot_losses(stats):
    """
    Plot the training and validation losses from the training statistics.
    """
    train_losses = stats['train_loss']
    val_losses = stats['val_loss']
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('losses.png')

def load_data(filepath):
    """Load the dataset from a file."""
    return torch.load(filepath)

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If best_model, saves separately"""
    torch.save(state, filename)

def get_batch(x, t_eval, batch_step):
    n_samples, n_points, input_dim = x.shape
    N = n_samples

    # Using torch to generate indices
    n_ids = torch.arange(N)  # equivalent to np.arange(N)
    # Randomly select starting points for each trajectory
    p_ids = torch.randint(0, n_points - batch_step, (N,))  # replace np.random.choice

    batch_x0 = x[n_ids, p_ids].reshape([N, 1, input_dim])
    batch_step += 1
    batch_t = t_eval[:batch_step]
    batch_x = torch.stack([x[n_ids, p_ids + i] for i in range(batch_step)], dim=0).reshape([batch_step, N, 1, input_dim])

    return batch_x0, batch_t, batch_x

def arrange(x, t_eval):
    n_samples, n_points, input_dim = x.shape

    # Using torch to generate indices
    n_ids = torch.arange(n_samples)  # equivalent to np.arange
    p_ids = torch.zeros(n_samples, dtype=torch.int64)  # replace np.array with zero-initialized tensor

    batch_x0 = x[n_ids, p_ids].reshape([n_samples, 1, input_dim])
    batch_t = t_eval
    batch_x = torch.stack([x[n_ids, p_ids + i] for i in range(n_points)], dim=0).reshape([n_points, n_samples, 1, input_dim])

    return batch_x0, batch_t, batch_x

class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super(ODEFunc, self).__init__()
        self.input_dim = input_dim
        self.sigma = nn.Parameter(torch.tensor([1e-1]))
        self.a = nn.Parameter(torch.ones(input_dim) * 1e-1)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            # nn.Tanh(),
            # nn.Linear(20, 20),
            # nn.Tanh(),
            # nn.Linear(20, 20),
            # nn.Tanh(),
            # nn.Linear(20, 20),
            # nn.Tanh(),
            # nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, input_dim)  # Output: [du/dx1, du/dx2]
        )


    def sampling_x0(self, x0):
        n, dammy, d = x0.shape
        return (x0 + torch.sqrt(torch.stack([self.a**2]*n).reshape([n,1,d]))
                * (torch.normal(0,1, size=(x0.shape[0],1,x0.shape[2]))))

    def neg_loglike(self, batch_x, pred_x):
        n_samples, n_points, dammy, input_dim = batch_x.shape
        likelihood = ( (-(pred_x-batch_x)**2/self.sigma**2/2).nansum()
                    - torch.log(self.sigma**2)/2*n_samples*n_points*input_dim)
        return -likelihood
    
    def forward(self, t, x):
        return self.network(x)

def plot_scatter_with_styles(batch_ys, pred_x):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Define a color map or list of colors
    colors = plt.cm.viridis(np.linspace(0, 1, 10))  # Creates 10 colors using the viridis colormap
    
    for i in range(10):
        # Scatter plot for batch_ys
        ax.plot(batch_ys[:, i, 0, 0].detach().numpy(), batch_ys[:, i, 0, 1].detach().numpy(), 
                color=colors[i])
        
        # Scatter plot for pred_x, using plot to create dashed style
        ax.plot(pred_x[:, i, 0, 0].detach().numpy(), pred_x[:, i, 0, 1].detach().numpy(), 
                linestyle='--', marker='o', color=colors[i])
    
    # Adding legend and labels might make the plot clearer
    ax.legend()
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Comparison of Batch and Predicted Coordinates')
    plt.show()

def train(model, train_data, val_data, learning_rate, batch_time, total_steps):
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    stats = defaultdict(list)
    min_val_loss = 1e+10
    t_eval = train_data['t'].clone().detach()
    batch_step = int(((len(t_eval)-1) / t_eval[-1]).item() * batch_time)

    for step in range(total_steps+1):
        # train step
        batch_y0, batch_t, batch_ys = get_batch(train_data['yts'], t_eval, batch_step)
        s_batch_x0 = model.sampling_x0(batch_y0)
        # model.sampling_epsilon_f()
        # pred_x = odeint(model, s_batch_x0, batch_t, method='dopri5', atol=1e-8, rtol=1e-8)
        pred_x = odeint(model, s_batch_x0, batch_t, method='fehlberg2', atol=1e-4, rtol=1e-4)
        loss = model.neg_loglike(batch_ys, pred_x)
        loss.backward(); optim.step(); optim.zero_grad()
        train_loss = loss.detach().item() / batch_y0.shape[0] / batch_t.shape[0]
        # run validation data
        with torch.no_grad():
            batch_y0, batch_t, batch_ys = arrange(val_data['yts'], t_eval)
            s_batch_x0 = model.sampling_x0(batch_y0)
            # pred_val_x = odeint(model, s_batch_x0, t_eval, method='dopri5', atol=1e-8, rtol=1e-8)
            pred_val_x = odeint(model, s_batch_x0, t_eval, method='fehlberg2', atol=1e-4, rtol=1e-4)
            val_loss = model.neg_loglike(batch_ys, pred_val_x)
            val_loss = val_loss.item() / batch_y0.shape[0] / t_eval.shape[0]
        # logging
        stats['train_loss'].append(train_loss)
        stats['val_loss'].append(val_loss)
        if step % 100 == 0:
            print(f"step {step}, train_loss {train_loss:.4e}, val_loss {val_loss:.4e}")

        if val_loss < min_val_loss:
            best_model = copy.deepcopy(model)
            min_val_loss = val_loss; best_train_loss = train_loss
            best_step = step
            # save it
            save_checkpoint({
                'step': step,
                'state_dict': model.state_dict(),
                'optim_dict': optim.state_dict(),
                'stats': stats,
                'best_train_loss': best_train_loss,
                'min_val_loss': min_val_loss,
                'best_step': best_step
            }, filename='nerualode.pth.tar')
    
    plot_scatter_with_styles(batch_ys, pred_x)

    # Plot pred_x.shape, batch_ys.shape
    return best_model, optim, stats, best_train_loss, min_val_loss, best_step

if __name__ == "__main__":
    # Load train and validation data
    train_data = load_data('pendulum_train_trajectories.pth')
    test_data = load_data('pendulum_test_trajectories.pth')

    input_dim = 2
    learning_rate = 1e-3
    batch_time = 1
    total_steps = 1000

    # Initialize the model
    model = ODEFunc(input_dim)

    # Learning
    t0 = time.time()
    best_model, optim, stats, train_loss, val_loss, step = train(model, train_data, test_data, learning_rate, batch_time, total_steps)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f} s")

    # Visualize results
    filepath = 'nerualode.pth.tar'  # Path to the saved model checkpoint
    checkpoint = load_checkpoint(filepath)
    
    # Assuming 'stats' is a dictionary containing lists of loss values
    stats = checkpoint['stats']
    plot_losses(stats)