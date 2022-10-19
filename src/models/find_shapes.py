import numpy as np
import torch


def linear_shape(input_shape, out_features):
    """
    Takes the given input shape fed into torch linear layer with provided parameters and computes the shape of the
    output.
    Parameters
    ----------
    input_shape: tuple containing shape of input data
    Check PyTorch docs for other parameter definitions: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Returns
    Tuple containing output shape
    -------
    """
    output_shape = input_shape
    output_shape[-1] = out_features
    return output_shape


def conv1d_shape(input_shape, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    """
    Takes the given input shape fed into torch Conv1d layer with provided parameters and computes the shape of the
    output. Expects unbatched data, so input_shape should be 2D.
    Parameters
    ----------
    input_shape: tuple containing shape of input data
    Check PyTorch docs for other parameter definitions: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    Returns
    -------
    Tuple containing output shape
    """
    l_in = input_shape[-1]
    l_out = np.floor(
        (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )
    output_shape = (out_channels, l_out)
    return output_shape


def convtranspose1d_shape(input_shape, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
    """
    Takes the given input shape fed into torch Conv1d layer with provided parameters and computes the shape of the
    output. Expects unbatched data, so input_shape should be 2D.
    Parameters
    ----------
    input_shape: tuple containing shape of input data
    Check PyTorch docs for other parameter definitions: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    Returns
    -------
    Tuple containing output shape
    """
    l_in = input_shape[-1]
    l_out = (l_in + 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    output_shape = (out_channels, l_out)
    return output_shape


def module_shape(input_shape, module, module_params):
    """
    Parameters
    ----------
    input_shape: tuple containing shape of input data, should be (num_channels, num_timesteps)
    module: class for module
    module_params: parameters passed to module, e.g. parameters for each layer
    Returns
    -------
    """
    test_net = module(*module_params)
    test_x = torch.ones(input_shape)
    test_x_hat = test_net(test_x)
    return test_x_hat.shape
