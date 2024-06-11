import torch
import torch.nn as nn


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

        loss = (
            targets * (1 + (- inputs).exp()).log() +
            (1 - targets) * (1 + inputs.exp()).log()
        )
        return loss


class BirdLoss(nn.Module):
    """
    Custom loss function for the problem.
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
        self.weighted = config.get("weighted", False)
        self.mask_secondary = config.get("mask_secondary", False)

        if config["name"] == "bce":
            if self.eps:
                self.loss = SmoothBCEWithLogitsLoss(eps=self.eps)
            else:
                self.loss = nn.BCEWithLogitsLoss(reduction="none")
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

    def forward(self, pred, y, secondary_mask=None, w=None):
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

        if self.mask_secondary:
            assert len(loss.size()) == 2, "Do not average per class before masking"
            loss *= (1 - secondary_mask)

        if len(loss.size()) == 2:
            loss = loss.mean(-1)

        if self.weighted and w is not None:
            loss = (loss * w).sum() / w.sum()
        else:
            loss = loss.mean()

        return loss
