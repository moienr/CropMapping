import numpy as np
import matplotlib.pyplot as plt


def plot_train_test_losses(train_losses:np.array, test_losses:np.array, title="Train Test Loss",
                           x_label="Epochs", y_label="Bianry Cross Entropy Loss",
                           min_max_bounds= True,
                           tight_x_lim = True, y_lim=None,
                           train_legend = "Train", test_legend = "Test",
                           save_path=None)->None:
    """
    This function takes in train and test losses as inputs and plots them using matplotlib.

    Parameters:
    ---
    train_losses (numpy array): Array of train losses for each epoch. The shape of the array should be (num_runs, num_epochs)
    test_losses (numpy array): Array of test losses for each epoch. The shape of the array should be (num_runs, num_epochs)
    title (str): Title of the plot (default is "Train Test Loss")
    x_label (str): Label for the x-axis (default is "Epochs")
    y_label (str): Label for the y-axis (default is "RMSE")
    min_max_bounds (bool): If True, the plot shows minimum and maximum values of losses, if False, the plot shows mean and standard deviation of losses (default is False)
    tight_x_lim (bool): If True, the x-axis limits are set to (0, num_epochs), if False, the x-axis limits are set automatically by matplotlib (default is True)
    y_lim (tuple): Limits for the y-axis (default is None)
    save_path (str): If provided, saves the plot at the given path (default is None)

    Returns:
    ---
    None

    Example Usage:
    ---
    plot_train_test_losses(train_losses, test_losses, title="Train Test Losses", x_label="Epochs", y_label="RMSE")
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 12
    mean_train_losses = np.mean(train_losses, axis=0)
    std_train_losses = np.std(train_losses, axis=0)
    mean_test_losses = np.mean(test_losses, axis=0)
    std_test_losses = np.std(test_losses, axis=0)
    if min_max_bounds:
        lower_train_losses = np.min(train_losses, axis=0)
        upper_train_losses = np.max(train_losses, axis=0)
        lower_test_losses = np.min(test_losses, axis=0)
        upper_test_losses = np.max(test_losses, axis=0)
    else:
        lower_train_losses = mean_train_losses - std_train_losses
        upper_train_losses = mean_train_losses + std_train_losses
        lower_test_losses = mean_test_losses - std_test_losses
        upper_test_losses = mean_test_losses + std_test_losses

    x_range = range(1, len(mean_train_losses) + 1)
    
    plt.plot(x_range ,mean_train_losses, color='#33a9a5', linewidth=2, label=train_legend)
    plt.fill_between(x_range, lower_train_losses, upper_train_losses, alpha=0.2, color='#33a9a5', edgecolor='none')

    plt.plot(x_range ,mean_test_losses, color='#f27085', linewidth=2, label=test_legend)
    plt.fill_between(x_range, lower_test_losses, upper_test_losses, alpha=0.2, color='#f27085', edgecolor='none')

    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    if tight_x_lim:
        plt.xlim(1, train_losses.shape[1])
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())
def convert2uint8(x):
    return (x * 255).astype(np.uint8)

def display_images(array1, array2, names, title, figsize = (10,5), savepath=None):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for ax, array, name in zip(axs, [array1, array2], names):
        ax.imshow(array)
        ax.set_title(name)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    fig.suptitle(title)
    # tighten the plot
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    
    
import matplotlib.pyplot as plt

def plot_s2_img(s2_img, n, m):
    """
    Plot an image for each depth of the s2_img tensor, by plotting the first 3 channels as an RGB image.
    
    Args:
    - s2_img: a tensor of shape (D, C, H, W)
    - n: number of rows in the subplot
    - m: number of columns in the subplot
    
    Returns:
    - None
    """
    # Move the tensor to the CPU and detach it
    s2_img = s2_img.cpu().detach()
    
    # Permute the tensor to have shape (D, H, W, C)
    s2_img = s2_img.permute(1, 2, 3, 0)
    
    # Create a new figure
    fig = plt.figure(figsize=(m*5, n*5))
    
    # Loop over the depths and plot an image for each depth
    for d in range(s2_img.shape[0]):
        # Extract the first 3 channels as an RGB image
        rgb_img = s2_img[d, :, :, :3]
        # reveser rgb channels
        rgb_img = rgb_img[:, :, [2, 1, 0]]
        
        # Plot the RGB image in a subplot
        ax = fig.add_subplot(n, m, d+1)
        ax.imshow(rgb_img)
        ax.set_title(f"Depth {d}")
    
    # Show the plot
    plt.show()
    

def plot_output_crop_map(output, crop_map, colorbar=False):
    """
    Plot the model output and crop map side by side for each band
    
    Args:
    - output: the model output tensor of shape (21, 64, 64)
    - crop_map: the crop map tensor of shape (21, 64, 64)
    - colorbar: whether to include color bar in the subplots (default: False)
    
    Returns:
    - None
    """
    # Loop over the bands
    for i in range(output.shape[0]):
        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the model output in the first subplot
        im1 = axs[0].imshow(output[i], cmap='gray')
        axs[0].set_title(f'Band {i+1} - Model Output')

        # Plot the crop map in the second subplot
        im2 = axs[1].imshow(crop_map[i], cmap='gray')
        axs[1].set_title(f'Band {i+1} - Crop Map')

        # Add color bar to subplots if colorbar is True
        if colorbar:
            fig.colorbar(im1, ax=axs[0])
            fig.colorbar(im2, ax=axs[1])

        # Show the plot
        plt.show()

if __name__ == "__main__":
    train_losses = np.random.random((10, 100)) * np.geomspace(100, 1, num=100, endpoint=True)  /100 
    test_losses = np.random.random((10, 100)) * np.geomspace(100, 1, num=100, endpoint=True)  /100 + np.linspace(.1, 0, num=100, endpoint=True) + 0.05

    plot_train_test_losses(train_losses, test_losses,y_lim=[0,1])
