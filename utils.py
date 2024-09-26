import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_pusht_images_sequnece(images, pos=None, action=None, title=None, rows=4, cols=4, softmax=False):
    """
    Visualize a batch of 16 images with dimensions 16 x 3 x 96 x 96.

    Args:
    - images (numpy array): A 4D numpy array of shape (16, 3, 96, 96)
                            representing 16 images with 3 channels (RGB).
    - rows (int): Number of rows in the grid (default is 4).
    - cols (int): Number of columns in the grid (default is 4).
    """
    # assert images.shape == (16, 3, 96, 96), "Input images must be of shape (16, 3, 96, 96)"
    # if pos is not None:
    #     print(pos.shape)
    #     images[:,:,pos[:,0],pos[:,1]]==0.

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # images = torch.zeros(16,3,96,96)
    # action = None
    for i, ax in enumerate(axes.flat):
        # Transpose the image from (3, 96, 96) to (96, 96, 3) for plotting
        image = images[i]
        flatten = image.view(image.shape[0],-1)
        softmax_flatten = torch.softmax(flatten,dim=1)
        softmax_result = softmax_flatten.view_as(image)
        if softmax:
            image = softmax_result
            print('softmax')
        img = np.transpose(image, (1, 2, 0))
        if pos is not None:
            x1, y1 = pos[i]
            circle = plt.Circle((x1, y1), radius=5, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)
        if action is not None:
            x2, y2 = action[i]
            ax.plot(x2, y2, marker='x', color='white', markersize=10, markeredgewidth=2)
        # Clip the pixel values between 0 and 1 if necessary
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('step k+{}'.format(i))

    if title is not None:
        fig.suptitle(title)
    plt.show()

# Example usage:
# Assuming `images` is a numpy array with shape (16, 3, 96, 96)
# visualize_images(images)
# import torch

# visualize_pusht_images_sequnece(torch.rand(16,3,96,96))


def pos2pixel(pos):
    pix = (pos/512)*96
    pix = pix.to(torch.long)
    # print('innnnnnnt')
    return pix

def pixel2pos(pix):
    pos = pix.to(torch.float)
    pos = (pix/96)*512
    return pos

def pixel2map(pix,h=96, w=96):
    # pix shape (n, 2)
    length = pix.shape[0]
    row_index = torch.arange(length).to(pix.device)
    one_hot_map = torch.zeros(length, h*w).to(pix.device)
    category = pix[:, 0]*w + pix[:, 1]
    # print(category)
    # print(one_hot_map.shape)
    one_hot_map[row_index, category.reshape(-1)] = 1.
    # print(torch.argmax(one_hot_map[1,:]))
    return one_hot_map