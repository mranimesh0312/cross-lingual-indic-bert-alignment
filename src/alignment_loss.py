from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cosine_alignment_loss(
    forward_hidden: torch.Tensor,
    forward_mask: torch.Tensor,
    reverse_hidden: torch.Tensor,
    reverse_mask: torch.Tensor,
) -> torch.Tensor:
    forward_sentence = mean_pool(forward_hidden, forward_mask)
    reverse_sentence = mean_pool(reverse_hidden, reverse_mask)
    labels = torch.ones(forward_sentence.size(0), device=forward_sentence.device)
    return F.cosine_embedding_loss(forward_sentence, reverse_sentence, labels)


def mse_alignment_loss(
    forward_hidden: torch.Tensor,
    forward_mask: torch.Tensor,
    reverse_hidden: torch.Tensor,
    reverse_mask: torch.Tensor,
) -> torch.Tensor:
    forward_sentence = mean_pool(forward_hidden, forward_mask)
    reverse_sentence = mean_pool(reverse_hidden, reverse_mask)
    return F.mse_loss(forward_sentence, reverse_sentence)

