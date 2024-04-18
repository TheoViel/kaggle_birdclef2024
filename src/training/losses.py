import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class FocalLossBCE(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "mean",
        bce_weight: float = 1.0,
        focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(inputs, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss


class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """

    def __init__(self, eps=0.0):
        """
        Constructor.
        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Computes the loss.
        Args:
            inputs (torch tensor [bs x n]): Predictions.
            targets (torch tensor [bs x n] or [bs]): Targets.
        Returns:
            torch tensor [bs]: Loss values.
        """
        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)

        if self.eps > 0:
            n_class = inputs.size(1)
            targets = targets * (1 - self.eps) + (1 - targets) * self.eps / (
                n_class - 1
            )

        loss = -targets * F.log_softmax(inputs, dim=1)
        # loss = loss.sum(-1)
        return loss


class SmoothBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss loss with label smoothing.
    """

    def __init__(self, eps=0.0):
        """
        Constructor.
        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
        """
        super(SmoothBCEWithLogitsLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Computes the loss.
        Args:
            inputs (torch tensor [bs x n]): Predictions.
            targets (torch tensor [bs x n]): Targets.
        Returns:
            torch tensor [bs]: Loss values.
        """
        assert inputs.size() == targets.size()

        if self.eps > 0:
            targets = torch.clamp(targets, self.eps, 1 - self.eps)

        # loss = - (
        #     targets * inputs.sigmoid().log() +
        #     (1 - targets) * (1 - inputs.sigmoid()).log()
        # )
        loss = (
            targets * (1 + (- inputs).exp()).log() +
            (1 - targets) * (1 + inputs.exp()).log()
        )
        return loss


class BirdLoss(nn.Module):
    """
    Custom loss function for the problem.

    Attributes:
        config (dict): Configuration parameters for the loss.
        device (str): Device to perform loss computations (e.g., "cuda" or "cpu").
        eps (float): Smoothing factor for the primary loss.
        loss (torch.nn.Module): Loss function for primary predictions.
        loss_aux (torch.nn.Module): Loss function for auxiliary predictions.
    """

    def __init__(self, config, device="cuda"):
        """
        Constructor for the AbdomenLoss class.

        Args:
            config (dict): Configuration parameters for the loss.
            device (str, optional): Device to perform loss computations. Defaults to "cuda".
        """
        super().__init__()
        self.config = config
        self.device = device

        self.eps = config.get("smoothing", 0)
        self.top_k = config.get("top_k", 0)

        if config["name"] == "bce":
            if self.eps:
                self.loss = SmoothBCEWithLogitsLoss(eps=self.eps)
            else:
                self.loss = nn.BCEWithLogitsLoss(reduction="none")

        elif config["name"] == "ce":
            self.loss = SmoothCrossEntropyLoss(eps=self.eps)
        elif config["name"] == "focal":
            self.loss = FocalLoss(eps=self.eps)
        else:
            raise NotImplementedError

    def prepare(self, pred, y):
        """
        Prepares the predictions and targets for loss computation.

        Args:
            pred (torch.Tensor): Main predictions.
            y (torch.Tensor): Main targets.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Prepared predictions and targets.
        """
        if self.config["name"] == "ce":
            y = y.squeeze(1)
        elif self.config["name"] in ["bce"]:
            y = y.float()
            pred = pred.float().view(y.size())
        else:
            pass

        return pred, y

    def top_k_mask(self, pred, y):
        weights = torch.ones_like(pred, device=pred.device)
        # Zero top_k preds weight
        weights.scatter_(1, torch.topk(pred, k=self.top_k).indices, torch.zeros_like(pred))
        # Ensure positive labels are weighted
        weights = torch.clamp(weights + (y > 0).float(), 0., 1.)
        return weights

    def forward(self, pred, y):
        """
        Computes the loss.

        Args:
            pred (torch.Tensor): Main predictions.
            y (torch.Tensor): Main targets.

        Returns:
            torch.Tensor: Loss value.
        """
        pred, y = self.prepare(pred, y)
        loss = self.loss(pred, y)

        # print(loss.size())

        if self.top_k:
            mask = self.top_k_mask(pred, y)
            loss *= mask

        # if len(loss.size()) > 2:
        #     loss = loss.sum(-1)

        return loss.mean()
