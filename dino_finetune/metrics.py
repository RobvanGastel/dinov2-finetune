from typing import Optional

import torch


def compute_iou_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int | None] = None,
    eps: float = 1e-6,
) -> float:
    """Compute the Intersection over Union metric for the predictions and labels.

    Args:
        y_hat (torch.Tensor): The prediction of dimensions (B, C, H, W), C being
            equal to the number of classes.
        y (torch.Tensor): The label for the prediction of dimensions (B, H, W)
        ignore_index (int | None, optional): ignore label to omit predictions in
            given region.
        eps (float, optional): To smooth the division and prevent division
        by zero. Defaults to 1e-6.

    Returns:
        float: The mean IoU
    """

    y_hat = torch.argmax(y_hat, dim=1)
    y_hat = y_hat.int()
    y = y.int()

    if ignore_index is not None:
        mask = y != ignore_index
        y_hat = y_hat * mask
        y = y * mask

    intersection = (y_hat & y).float().sum((1, 2))
    union = (y_hat | y).float().sum((1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean()
