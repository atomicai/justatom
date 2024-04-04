from typing import Optional, Union
from numbers import Real
from loguru import logger
import warnings
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from collections import OrderedDict

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable

# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py
# adapted from:
# https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/focal.html


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]` for one-vs-others mode (weight of negative class)
                        or :math:`\alpha_i \in \R`
                        vector of weights for each class (analogous to weight argument for CrossEntropyLoss)
        gamma (float): Focusing parameter :math:`\gamma >= 0`. When 0 is equal to CrossEntropyLoss
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’.
         ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
                in the output, uses geometric mean if alpha set to list of weights
         ‘sum’: the output will be summed. Default: ‘none’.
        ignore_index (Optional[int]): specifies indexes that are ignored during loss calculation
         (identical to PyTorch's CrossEntropyLoss 'ignore_index' parameter). Default: -100

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> C = 5  # num_classes
        >>> N = 1 # num_examples
        >>> loss = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(N, C, requires_grad=True)
        >>> target = torch.empty(N, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[Union[float, Iterable]] = None,
        gamma: Real = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, float) and not isinstance(alpha, Iterable):
            raise ValueError(f"alpha value should be None, float value or list of real values. Got: {type(alpha)}")
        self.alpha: Optional[Union[float, torch.Tensor]] = (
            alpha if alpha is None or isinstance(alpha, float) else torch.FloatTensor(alpha)
        )
        if isinstance(alpha, float) and not 0.0 <= alpha <= 1.0:
            warnings.warn("[Focal Loss] alpha value is to high must be between [0, 1]")

        self.gamma: Real = gamma
        self.reduction: str = reduction
        self.ignore_index: int = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))
        if input.shape[0] != target.shape[0]:
            raise ValueError(
                f"First dimension of inputs and targets should be same shape. Got: {input.shape} and {target.shape}"
            )
        if len(input.shape) != 2 or len(target.shape) != 1:
            raise ValueError(f"input tensors should be of shape (N, C) and (N,). Got: {input.shape} and {target.shape}")
        if input.device != target.device:
            raise ValueError("input and target must be in the same device. Got: {}".format(input.device, target.device))

        # filter labels
        target = target.type(torch.long)
        input_mask = target != self.ignore_index
        target = target[input_mask]
        input = input[input_mask]
        # compute softmax over the classes axis
        pt = F.softmax(input, dim=1)
        logpt = F.log_softmax(input, dim=1)

        # compute focal loss
        pt = pt.gather(1, target.unsqueeze(-1)).squeeze()
        logpt = logpt.gather(1, target.unsqueeze(-1)).squeeze()
        focal_loss = -1 * (1 - pt) ** self.gamma * logpt

        weights = torch.ones_like(focal_loss, dtype=focal_loss.dtype, device=focal_loss.device)
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha = torch.tensor(self.alpha, device=input.device)
                weights = torch.where(target > 0, 1 - alpha, alpha)
            elif torch.is_tensor(self.alpha):
                alpha = self.alpha.to(input.device)
                weights = alpha.gather(0, target)

        tmp_loss = focal_loss * weights
        if self.reduction == "none":
            loss = tmp_loss
        elif self.reduction == "mean":
            loss = tmp_loss.sum() / weights.sum() if torch.is_tensor(self.alpha) else torch.mean(tmp_loss)
        elif self.reduction == "sum":
            loss = tmp_loss.sum()
        else:
            raise NotImplementedError("Invalid reduction mode: {}".format(self.reduction))
        return loss


class MultiMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MultiMarginLoss, self).__init__()
        self.loss = torch.nn.MultiMarginLoss(margin=margin)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_soft = F.softmax(input, dim=1)
        return self.loss(input=input_soft, target=target)


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N,)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = DiceLoss()
        >>> input = torch.randn(2, N, requires_grad=True)
        >>> target = torch.empty(2, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(
        self,
        gamma: int = 0,
        scale: float = 1.0,
        reduction: Optional[str] = "mean",
        ignore_index: int = -100,
        eps: float = 1e-6,
        smooth: float = 0,
    ) -> None:
        super(DiceLoss, self).__init__()
        self.gamma: int = gamma
        self.scale: float = scale
        self.reduction: Optional[str] = reduction
        self.ignore_index: int = ignore_index
        self.eps: float = eps
        self.smooth: float = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 2:
            if input.shape[0] != target.shape[0]:
                raise ValueError(
                    "number of elements in input and target shapes must be the same. Got: {}".format(
                        input.shape, input.shape
                    )
                )
        else:
            raise ValueError("Invalid input shape, we expect or NxC. Got: {}".format(input.shape))
        if not input.device == target.device:
            raise ValueError("input and target must be in the same device. Got: {}".format(input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        input_soft = self.scale * ((1 - input_soft) ** self.gamma) * input_soft

        # filter labels
        target = target.type(torch.long)
        input_mask = target != self.ignore_index

        target = target[input_mask]
        input_soft = input_soft[input_mask]

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input_soft.shape[-1]).to(input.device).type(input_soft.dtype)

        # compute the actual dice score
        intersection = torch.sum(input_soft * target_one_hot, dim=-1)
        cardinality = torch.sum(input_soft + target_one_hot, dim=-1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.eps + self.smooth)
        dice_loss = 1.0 - dice_score

        if self.reduction is None or self.reduction == "none":
            return dice_loss
        elif self.reduction == "mean":
            return torch.mean(dice_loss)
        elif self.reduction == "sum":
            return torch.sum(dice_loss)
        else:
            raise NotImplementedError("Invalid reduction mode: {}".format(self.reduction))


class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coeficient loss.

    According to [1], we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N,)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: int = 0,
        scale: float = 1.0,
        reduction: Optional[str] = "mean",
        ignore_index: int = -100,
        eps: float = 1e-6,
        smooth: float = 0,
    ) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: int = gamma
        self.scale: float = scale
        self.reduction: Optional[str] = reduction
        self.ignore_index: int = ignore_index
        self.eps: float = eps
        self.smooth: float = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 2:
            if input.shape[0] != target.shape[0]:
                raise ValueError(
                    "number of elements in input and target shapes must be the same. Got: {}".format(
                        input.shape, input.shape
                    )
                )
        else:
            raise ValueError("Invalid input shape, we expect or NxC. Got: {}".format(input.shape))
        if not input.device == target.device:
            raise ValueError("input and target must be in the same device. Got: {}".format(input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        input_soft = self.scale * ((1 - input_soft) ** self.gamma) * input_soft

        # filter labels
        target = target.type(torch.long)
        input_mask = target != self.ignore_index

        target = target[input_mask]
        input_soft = input_soft[input_mask]

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).to(input.device).type(input_soft.dtype)

        # compute the actual dice score
        intersection = torch.sum(input_soft * target_one_hot, -1)
        fps = torch.sum(input_soft * (1.0 - target_one_hot), -1)
        fns = torch.sum((1.0 - input_soft) * target_one_hot, -1)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = (numerator + self.smooth) / (denominator + self.eps + self.smooth)
        tversky_loss = 1.0 - tversky_loss

        if self.reduction is None or self.reduction == "none":
            return tversky_loss
        elif self.reduction == "mean":
            return torch.mean(tversky_loss)
        elif self.reduction == "sum":
            return torch.sum(tversky_loss)
        else:
            raise NotImplementedError("Invalid reduction mode: {}".format(self.reduction))


def pdist(v):
    dist = torch.norm(v[:, None] - v, dim=2, p=2)
    return dist


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, sample=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.sample = sample

    def forward(self, inputs, targets):
        """
        inputs: torch.Tensor. n x embedding_dim
        targets: torch.Tensor n x 1.
        y_i = targets[i] indicate which group x_i belongs to.
        """
        n = inputs.size(0)  # batch_size samples
        _device = inputs.device
        # pairwise distances
        dist = pdist(inputs)  # The same as taking inputs @ inputs.T => n x n

        # find the hardest positive and negative
        mask_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_neg = ~mask_pos
        mask_pos[torch.eye(n).bool()] = 0
        if self.sample:
            # weighted sample pos and negative to avoid outliers causing collapse
            posw = (dist + 1e-12) * mask_pos.float()
            posi = torch.multinomial(posw, 1)
            dist_p = dist.gather(0, posi.view(1, -1))
            # There is likely a much better way of sampling negatives in proportion their difficulty, based on distance
            # this was a quick hack that ended up working better for some datasets than hard negative
            negw = (1 / (dist + 1e-12)) * mask_neg.float()
            negi = torch.multinomial(negw, 1)
            dist_n = dist.gather(0, negi.view(1, -1))
        else:
            # hard negative
            # 1. Fill with -inf [i,j] matrix. (the same shape as dist)
            ninf = torch.ones_like(dist).to(_device) * float("-inf")  # need to move to device
            # 2. Positive max distance calculation within the same groups
            #    dist_p.shape = batch_size.
            #    dist_p[i] => max(distance(sample_i, sample_j) : sample_j and sample_i belong to the same group))
            dist_p = torch.max(dist * mask_pos.float(), dim=1)[0]
            # 3.
            nindex = torch.max(torch.where(mask_neg, -dist, ninf), dim=1)[1]
            dist_n = dist.gather(0, nindex.unsqueeze(0))

        # calc loss
        diff = dist_p - dist_n
        if isinstance(self.margin, str) and self.margin == "soft":
            diff = F.softplus(diff)
        else:
            diff = torch.clamp(diff + self.margin, min=0.0)
        loss = diff.mean()

        # calculate metrics, no impact on loss
        metrics = OrderedDict()
        with torch.no_grad():
            _, top_idx = torch.topk(dist, k=2, largest=False)
            top_idx = top_idx[:, 1:]
            flat_idx = top_idx.squeeze() + n * torch.arange(n, out=torch.LongTensor()).to(_device)
            top1_is_same = torch.take(mask_pos, flat_idx)
            metrics["prec"] = top1_is_same.float().mean().item()
            metrics["dist_acc"] = (dist_n > dist_p).float().mean().item()
            if not isinstance(self.margin, str):
                metrics["dist_sm"] = (dist_n > dist_p + self.margin).float().mean().item()
                metrics["nonzero_count"] = torch.nonzero(diff).size(0)
            metrics["dist_p"] = dist_p.mean().item()
            metrics["dist_n"] = dist_n.mean().item()
            metrics["rel_dist"] = ((dist_n - dist_p) / torch.max(dist_p, dist_n)).mean().item()

        return loss, metrics


class UMAPLoss(nn.Module):

    def __init__(self, negative_sample_rate=5):
        self.negative_sample_rate = negative_sample_rate

    def forward(self, embedding_to: torch.Tensor, embedding_from: torch.Tensor, _a, _b, _batch_size):
        assert embedding_to.device == embedding_from.device, logger.error(
            f"Device mismatch found. Device {embedding_to.device} != {embedding_from.device}"
        )
        device = embedding_to.device
        embedding_neg_to = embedding_to.repeat(self.negative_sample_rate, 1)
        repeat_neg = embedding_from.repeat(self.negative_sample_rate, 1)
        embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
        distance_embedding = torch.cat(
            ((embedding_to - embedding_from).norm(dim=1), (embedding_neg_to - embedding_neg_from).norm(dim=1)), dim=0
        )

        # convert probabilities to distances
        probabilities_distance = self.convert_distance_to_probability(distance_embedding, _a, _b)
        # set true probabilities based on negative sampling
        probabilities_graph = torch.cat(
            (torch.ones(_batch_size), torch.zeros(_batch_size * self.negative_sample_rate)),
            dim=0,
        )

        # compute cross entropy
        (attraction_loss, repellant_loss, ce_loss) = self.compute_cross_entropy(
            probabilities_graph.to(device),
            probabilities_distance.to(device),
        )
        loss = torch.mean(ce_loss)
        return loss

    def convert_distance_to_probability(self, distances, a=1.0, b=1.0):
        return -torch.log1p(a * distances ** (2 * b))

    def compute_cross_entropy(self, probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0):
        # cross entropy
        attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(probabilities_distance)
        repellant_term = (
            -(1.0 - probabilities_graph)
            * (torch.nn.functional.logsigmoid(probabilities_distance) - probabilities_distance)
            * repulsion_strength
        )

        # balance the expected losses between atrraction and repel
        CE = attraction_term + repellant_term
        return attraction_term, repellant_term, CE


def init_loss(device, weight: Optional = None, name: Optional[str] = None, **props):
    if weight and name is not None:
        logger.warning(
            "weight and name parametters are set at the same time"
            f"will use weighted cross entropy loss. To use {name} loss set weight to None"
        )
    if weight:
        loss_fct = CrossEntropyLoss(weight=torch.Tensor(weight).to(device))
    elif name:
        if name == "focal":
            loss_fct = FocalLoss(**props).to(device)
        elif name == "dice":
            loss_fct = DiceLoss(**props).to(device)
        elif name == "tversky":
            loss_fct = TverskyLoss(**props).to(device)
        elif name == "margin":
            loss_fct = MultiMarginLoss(**props).to(device)
        elif name == "triplet":
            loss_fct = TripletLoss(**props).to(device)
        elif name == "umap":
            loss_fct = UMAPLoss(**props).to(device)
        else:
            raise NotImplementedError(f"unknown {name} loss function")
    else:
        loss_fct = None

    return loss_fct


BCE_TYPE_LOSS = dict(focal=FocalLoss, dice=DiceLoss, tversky=TverskyLoss)


__all__ = ["init_loss", "BCE_TYPE_LOSS"]