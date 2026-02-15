import os
import random
from typing import BinaryIO, IO

import torch
from jaxtyping import Float, Int
import numpy.typing as npt
from torch import Tensor, nn
from collections.abc import Iterable

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[
    Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    bs = inputs.shape[0]
    x_scale = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum = torch.log(torch.exp(x_scale).sum(dim=-1, keepdim=True))
    return (log_sum - x_scale)[torch.arange(bs), targets].mean()

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6
    l2_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        l2_norm += torch.linalg.vector_norm(p.grad.data, ord=2) ** 2

    l2_norm = torch.sqrt(l2_norm)
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    check_point = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(check_point, out)


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    check_point = torch.load(src)
    model.load_state_dict(check_point["model"])
    optimizer.load_state_dict(check_point["optimizer"])
    return check_point["iteration"]

def get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    sl = len(dataset)
    x_list = []
    y_list = []
    for _ in range(batch_size):
        start = random.randint(0, sl - context_length - 1)
        x = torch.tensor(dataset[start:start + context_length], dtype=torch.long, device=device)
        y = torch.tensor(dataset[start + 1:start + context_length + 1], dtype=torch.long, device=device)
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list).to(device), torch.stack(y_list).to(device)