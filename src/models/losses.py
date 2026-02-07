"""Loss functions: FocalLoss, LabelSmoothingCE, MultiTaskLoss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default=2.0).
        alpha: Per-class weights tensor (optional).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits.
            targets: (B,) class indices.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def forward_mixup(
        self,
        logits: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Focal loss compatible with MixUp/CutMix."""
        return lam * self.forward(logits, targets_a) + (1 - lam) * self.forward(
            logits, targets_b
        )


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing.

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing).
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def forward_mixup(
        self,
        logits: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Label smoothing CE compatible with MixUp/CutMix."""
        return lam * self.forward(logits, targets_a) + (1 - lam) * self.forward(
            logits, targets_b
        )


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weighting (Kendall et al. 2018).

    L = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)

    Args:
        num_tasks: Number of tasks.
        task_names: Optional names for each task.
    """

    def __init__(self, num_tasks: int = 3, task_names: list = None):
        super().__init__()
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]

    def forward(self, losses: list) -> tuple:
        """
        Args:
            losses: List of individual task losses.

        Returns:
            total_loss: Weighted sum.
            loss_dict: Dictionary of individual and weighted losses.
        """
        total_loss = torch.tensor(0.0, device=losses[0].device)
        loss_dict = {}

        for i, (loss, name) in enumerate(zip(losses, self.task_names)):
            precision = torch.exp(-self.log_vars[i])
            weighted = precision * loss + self.log_vars[i]
            total_loss = total_loss + weighted
            loss_dict[f"{name}_raw"] = loss.item()
            loss_dict[f"{name}_weight"] = precision.item()
            loss_dict[f"{name}_logvar"] = self.log_vars[i].item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict
