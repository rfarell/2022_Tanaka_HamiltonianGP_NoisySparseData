import torch
from collections import defaultdict
from torchdiffeq import odeint
import time
import copy
from my_model import SSGP

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
        model.sampling_epsilon_f()
        # pred_x = odeint(model, s_batch_x0, batch_t, method='dopri5', atol=1e-8, rtol=1e-8)
        pred_x = odeint(model, s_batch_x0, batch_t, method='fehlberg2', atol=1e-4, rtol=1e-4)
        neg_loglike = model.neg_loglike(batch_ys, pred_x)
        KL_x0 = model.KL_x0(batch_y0.squeeze())
        KL_w = model.KL_w()
        loss = neg_loglike + KL_w + KL_x0
        loss.backward(); optim.step(); optim.zero_grad()
        train_loss = loss.detach().item() / batch_y0.shape[0] / batch_t.shape[0]
    
        # run validation data
        with torch.no_grad():
            batch_y0, batch_t, batch_ys = arrange(val_data['yts'], t_eval)
            s_batch_x0 = model.sampling_x0(batch_y0)
            model.mean_w()
            # pred_val_x = odeint(model, s_batch_x0, t_eval, method='dopri5', atol=1e-8, rtol=1e-8)
            pred_val_x = odeint(model, s_batch_x0, t_eval, method='fehlberg2', atol=1e-4, rtol=1e-4)
            val_neg_loglike = model.neg_loglike(batch_ys, pred_val_x)
            val_Kl_x0 = model.KL_x0(batch_y0.squeeze())
            val_Kl_w = model.KL_w()
            val_loss = val_neg_loglike + val_Kl_w + val_Kl_x0
            val_loss = val_neg_loglike.item() / batch_y0.shape[0] / t_eval.shape[0]
        # logging
        stats['train_loss'].append(train_loss)
        stats['train_kl_x0'].append(KL_x0.item())
        stats['train_kl_w'].append(KL_w.item())
        stats['train_neg_loglike'].append(neg_loglike.item() / batch_y0.shape[0] / batch_t.shape[0])
        stats['val_loss'].append(val_loss)
        stats['val_kl_x0'].append(val_Kl_x0.item())
        stats['val_kl_w'].append(val_Kl_w.item())
        stats['val_neg_loglike'].append(val_neg_loglike.item() / batch_y0.shape[0] / t_eval.shape[0])
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
            }, filename='best.pth.tar')
            
    return best_model, optim, stats, best_train_loss, min_val_loss, best_step

if __name__ == "__main__":
    # Load train and validation data
    train_data = load_data('pendulum_train_trajectories.pth')
    test_data = load_data('pendulum_test_trajectories.pth')

    input_dim = 2
    num_basis = 100
    friction = True
    K = 100
    learning_rate = 1e-3
    batch_time = 1
    total_steps = 1000

    # Initialize the model
    model = SSGP(input_dim, num_basis, friction, K)

    # Learning
    t0 = time.time()
    best_model, optim, stats, train_loss, val_loss, step = train(model, train_data, test_data, learning_rate, batch_time, total_steps)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f} s")
