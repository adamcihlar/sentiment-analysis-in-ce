from typing import List

import torch


def layer_wise_learning_rate(
    chronological_list_of_layers: List, base_lr, lr_decay=0.95, n_layers_following=1
):
    """
    Returns list of dictionaries containing parameters group (layer) and its
    learning rate.
    It expects that the final model has n_layers_following on top of these layers.
    """
    reversed_layers_iter = enumerate(reversed(chronological_list_of_layers))
    optimizer_params_list = list(
        reversed(
            [
                {
                    "params": l.parameters(),
                    "lr": base_lr * (lr_decay ** (i + n_layers_following)),
                }
                for i, l in reversed_layers_iter
            ]
        )
    )
    return optimizer_params_list


def inverted_sigmoid(probabilities: torch.Tensor):
    logits = torch.log(probabilities / (1 - probabilities))
    return logits


def to_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def get_log_prob_for_kl_div(probs: torch.Tensor):
    probs_ext = torch.concat((probs, to_cuda(torch.ones(probs.shape)) - probs), axis=1)
    log_probs_ext = torch.log(probs_ext)
    return log_probs_ext
