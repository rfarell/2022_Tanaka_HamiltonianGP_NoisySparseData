import torch
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    filepath = 'best.pth.tar'  # Path to the saved model checkpoint
    checkpoint = load_checkpoint(filepath)
    
    # Assuming 'stats' is a dictionary containing lists of loss values
    stats = checkpoint['stats']
    plot_losses(stats)
