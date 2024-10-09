# amr 2024
# Neural Cellular Automata Model and Utils
# adapted from: 
# blog - https://distill.pub/2020/growing-ca/
# code - https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb#scrollTo=zR6I1JONmWBb

import io
from PIL import Image
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from displayUtils import *

def load_image(url, max_size=TARGET_SIZE):
    """
    Load an image from a URL and resize it to a maximum size.

    Args:
        url (str): The URL of the image to load.
        max_size (int): The maximum size for the image's width and height.

    Returns:
        np.ndarray: The loaded and resized image as a NumPy array.
    """
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    # Premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img

def load_emoji(emoji):
    """
    Load an emoji image from the Noto Emoji GitHub repository.

    Args:
        emoji (str): The emoji character to load.

    Returns:
        np.ndarray: The loaded emoji image as a NumPy array.
    """
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true' % code
    return load_image(url)

def to_rgba(x):
    """
    Extract the RGBA channels from an image.

    Args:
        x (np.ndarray): The input image.

    Returns:
        np.ndarray: The RGBA channels of the image.
    """
    return x[..., :4]

def to_alpha(x):
    """
    Extract and clip the alpha channel from an image.

    Args:
        x (np.ndarray): The input image.

    Returns:
        np.ndarray: The clipped alpha channel.
    """
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
    """
    Convert an image with premultiplied alpha to RGB.

    Args:
        x (np.ndarray): The input image with premultiplied alpha.

    Returns:
        np.ndarray: The RGB channels of the image.
    """
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb

# TODO: adjust this to pytorch
def get_living_mask(x):
    """
    Get a mask indicating the living cells in the cellular automata.

    Args:
        x (np.ndarray): The input state of the cellular automata.

    Returns:
        np.ndarray: A mask indicating living cells.
    """
    alpha = x[:, :, :, 3:4]
    return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1

def make_seed(size, n=1):
    """
    Create a seed for the cellular automata.

    Args:
        size (int): The size of the seed.
        n (int): The number of seeds to create.

    Returns:
        np.ndarray: The created seed(s).
    """
    x = np.zeros([n, size, size, CHANNEL_N], np.float32)
    x[:, size // 2, size // 2, 3:] = 1.0
    return x

class NeuralCA(nn.Module):
    """
    A PyTorch implementation of a Cellular Automata Model.
    """

    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        """
        Initialize the NeuralCA.

        Args:
            channel_n (int): The number of channels in the model.
            fire_rate (float): The rate at which cells update.
        """
        super(NeuralCA, self).__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        self.dmodel = nn.Sequential(
            nn.Conv2d(self.channel_n, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, self.channel_n, kernel_size=1),
        )

    def perceive(self, x, angle=0.0):
        """
        Perceive the environment using convolutional filters.

        Args:
            x (torch.Tensor): The input tensor.
            angle (float): The angle for the perception filter.

        Returns:
            torch.Tensor: The perceived environment.
        """
        identify = torch.tensor([[0, 1, 0]], dtype=torch.float32)
        identify = torch.outer(identify, identify)
        dx = torch.outer(torch.tensor([1, 2, 1], dtype=torch.float32), torch.tensor([-1, 0, 1], dtype=torch.float32)) / 8.0
        dy = dx.T
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        kernel = torch.stack([identify, c * dx - s * dy, s * dx + c * dy], dim=-1).unsqueeze(2)
        kernel = kernel.repeat(1, 1, self.channel_n, 1)
        y = F.conv2d(x, kernel, padding=1, groups=self.channel_n)
        return y

    def forward(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        """
        Forward pass of the NeuralCA.

        Args:
            x (torch.Tensor): The input tensor.
            fire_rate (float, optional): The rate at which cells update.
            angle (float): The angle for the perception filter.
            step_size (float): The step size for updates.

        Returns:
            torch.Tensor: The updated state of the cellular automata.
        """
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.dmodel(y) * step_size
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = torch.rand_like(x[:, :1, :, :]) <= fire_rate
        x = x + dx * update_mask.float()

        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * life_mask.float()

# Note: You will need to adjust the `get_living_mask` function to work with PyTorch tensors.

# Original:

# class CAModel(tf.keras.Model):

#   def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
#     super().__init__()
#     self.channel_n = channel_n
#     self.fire_rate = fire_rate

#     self.dmodel = tf.keras.Sequential([
#           Conv2D(128, 1, activation=tf.nn.relu),
#           Conv2D(self.channel_n, 1, activation=None,
#               kernel_initializer=tf.zeros_initializer),
#     ])

#     self(tf.zeros([1, 3, 3, channel_n]))  # dummy call to build the model

#   @tf.function
#   def perceive(self, x, angle=0.0):
#     identify = np.float32([0, 1, 0])
#     identify = np.outer(identify, identify)
#     dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
#     dy = dx.T
#     c, s = tf.cos(angle), tf.sin(angle)
#     kernel = tf.stack([identify, c*dx-s*dy, s*dx+c*dy], -1)[:, :, None, :]
#     kernel = tf.repeat(kernel, self.channel_n, 2)
#     y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
#     return y

#   @tf.function
#   def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
#     pre_life_mask = get_living_mask(x)

#     y = self.perceive(x, angle)
#     dx = self.dmodel(y)*step_size
#     if fire_rate is None:
#       fire_rate = self.fire_rate
#     update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
#     x += dx * tf.cast(update_mask, tf.float32)

#     post_life_mask = get_living_mask(x)
#     life_mask = pre_life_mask & post_life_mask
#     return x * tf.cast(life_mask, tf.float32)


# CAModel().dmodel.summary()