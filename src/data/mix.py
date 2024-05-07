import torch
import torch.nn as nn


class Mixup(nn.Module):
    def __init__(self, p=1., alpha=0.4, additive=False, num_classes=1, **kwargs):
        """
        Mixup augmentation module.

        Args:
            alpha (float): Mixup interpolation parameter.
            additive (bool, optional): Whether to use additive mixup. Defaults to False.
            num_classes (int, optional): Number of classes. Defaults to 1.
        """
        super().__init__()
        self.beta_distribution = torch.distributions.Beta(alpha, alpha)
        self.additive = additive
        self.num_classes = num_classes
        self.p = p

    def forward(self, x, y=None, y_aux=None, w=None):
        """
        Forward pass of the Mixup module.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels for the main task.
            y_aux (torch.Tensor, optional): Target labels for the auxiliary task. Defaults to None.

        Returns:
            torch.Tensor: Mixed input data.
            torch.Tensor: Mixed target labels for the main task.
            torch.Tensor: Mixed target labels for the auxiliary task.
        """
        if self.p <= 0:
            # if not torch.rand(1).item() < self.p:
            return x, y, y_aux, w

        bs = x.shape[0]
        n_dims = len(x.shape)

        perm = torch.randperm(bs)
        perm = torch.where(torch.rand(bs) < self.p, perm, torch.arange(bs))

        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(x.device)

        # coeffs = torch.where(torch.rand(bs) < self.p, 0.2 + 0.8 * torch.rand(bs), 0)
        # coeffs = coeffs.to(x.device)
        # coeffs = torch.where(torch.rand(bs).to(coeffs.device) < self.p, coeffs, 1)

        if n_dims == 2:
            x = coeffs.view(-1, 1) * x + (1 - coeffs.view(-1, 1)) * x[perm]
        elif n_dims == 3:
            x = coeffs.view(-1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1)) * x[perm]
        elif n_dims == 4:
            x = coeffs.view(-1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1)) * x[perm]
        else:
            x = (
                coeffs.view(-1, 1, 1, 1, 1) * x
                + (1 - coeffs.view(-1, 1, 1, 1, 1)) * x[perm]
            )

        if y is not None:
            if self.num_classes > y.size(-1):  # One-hot
                y = (
                    torch.zeros(y.size(0), self.num_classes)
                    .to(y.device)
                    .scatter(1, y.view(-1, 1).long(), 1)
                )

            if self.additive:
                y = torch.cat([y.unsqueeze(0), y[perm].unsqueeze(0)], 0).amax(0)
            else:
                if len(y.shape) == 1:
                    y = coeffs * y + (1 - coeffs) * y[perm]
                else:
                    y = coeffs.view(-1, 1) * y + (1 - coeffs.view(-1, 1)) * y[perm]

        if y_aux is not None:
            y_aux = torch.cat([y_aux.unsqueeze(0), y_aux[perm].unsqueeze(0)], 0).amax(0)
            y_aux = torch.clamp(y_aux - y, 0, 1)  # Ensure primary label not in secondary

        if w is not None:
            w = coeffs * w + (1 - coeffs) * w[perm]
            assert w.sum() > 0, f'Weights: {w}'

        return x, y, y_aux, w
