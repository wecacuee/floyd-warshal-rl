import torch


def conv2d_output_size(in_shape, kernel_size, padding = 0, dilation = 1,
                       stride = 1):
    """ Compute output size form convolution layer

    >>> c1 = conv2d_output_size((84, 84), kernel_size = 8, stride = 4,
    ...                         padding = 1)
    >>> c1
    [20, 20]
    >>> c2 = conv2d_output_size(c1, kernel_size = 4, stride = 2)
    >>> c2
    [9, 9]
    >>> c3 = conv2d_output_size(c2, kernel_size = 3)
    >>> c3
    [7, 7]
    """
    in_shape = torch.as_tensor(in_shape, dtype=torch.float64)
    assert in_shape.dim() == 1 and in_shape.shape == (2,), "bad input shape"
    return torch.floor(
        (in_shape + 2 * padding - dilation * (kernel_size - 1) - 1) /
        stride + 1
    ).to(dtype=torch.int64).tolist()


def linscale(x, src_range, target_range):
    ss, se = src_range
    ts, te = target_range
    return (x - ss) / (se - ss) * (te - ts) + ts


def egreedy_prob_exp(step, start_eps = 0.5, end_eps = 0.001, nepisodes = None,
                     alpha = -20.0):
    """
    >>> egreedy_prob_exp(torch.tensor([0, 500, 1000], dtype=torch.float32),
    ...                  start_eps = 0.8, end_eps = 0.001,
    ...                  nepisodes = 1000,
    ...                  alpha = torch.log(torch.tensor(0.001 / 0.8)))
    tensor([0.8000, 0.0283, 0.0010])
    """
    assert nepisodes is not None, "nepisodes is required"
    step = torch.as_tensor(step, dtype=torch.float32)
    alpha = torch.as_tensor(alpha)
    # scale later
    nepisodes = step.new_tensor(nepisodes, dtype=step.dtype)
    step_clipped = torch.min(step, nepisodes)
    return linscale(
        torch.exp(alpha * step_clipped / nepisodes),
        (1, torch.exp(alpha)), (start_eps, end_eps))
