from typing import List, Optional
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from mmaction.registry import MODELS
from .base import BaseWeightedLoss


@MODELS.register_module()
class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        """
        logit: [N, num_cls]
        target: [N, ]
        """
        target = torch.argmax(target, dim=1)  # [N, num_cls] -> [N, ]

        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)
        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss


@MODELS.register_module()
class CoarseFocalLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probability distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        self.alpha = alpha
        self.gamma = gamma

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()

            prob_coarse = torch.empty((cls_score.shape[0], 7), device=cls_score.device)  # [N, 7]
            prob_fine = F.softmax(cls_score, dim=1)  # [N, 52]
            prob_coarse[:, 0] = torch.sum(prob_fine[:, :5], dim=1)
            prob_coarse[:, 1] = torch.sum(prob_fine[:, 5:11], dim=1)
            prob_coarse[:, 2] = torch.sum(prob_fine[:, 11:24], dim=1)
            prob_coarse[:, 3] = torch.sum(prob_fine[:, 24:32], dim=1)
            prob_coarse[:, 4] = torch.sum(prob_fine[:, 32:38], dim=1)
            prob_coarse[:, 5] = torch.sum(prob_fine[:, 38:48], dim=1)
            prob_coarse[:, 6] = torch.sum(prob_fine[:, 48:], dim=1)
            
            gt_coarse = torch.empty((cls_score.shape[0], 7), device=cls_score.device)  # [N, 7]
            gt_coarse[:, 0] = torch.sum(label[:, :5], dim=1)
            gt_coarse[:, 1] = torch.sum(label[:, 5:11], dim=1)
            gt_coarse[:, 2] = torch.sum(label[:, 11:24], dim=1)
            gt_coarse[:, 3] = torch.sum(label[:, 24:32], dim=1)
            gt_coarse[:, 4] = torch.sum(label[:, 32:38], dim=1)
            gt_coarse[:, 5] = torch.sum(label[:, 38:48], dim=1)
            gt_coarse[:, 6] = torch.sum(label[:, 48:], dim=1)
            gt_coarse = torch.argmax(gt_coarse, dim=1, keepdim=True)  # [N, 7]->[N, 1]

            prob = prob_coarse.gather(1, gt_coarse).view(-1)
            loss_coarse = -self.alpha * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob)
            loss_cls += torch.mean(loss_coarse)

        else:
            # calculate loss for hard label
            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            # loss_cls = F.cross_entropy(cls_score, label, **kwargs)
            label2 = label.clone()
            label_smooth_eps=0.1
            label = F.one_hot(label, num_classes=cls_score.shape[1])
            label = ((1 - label_smooth_eps) * label +
                      label_smooth_eps / cls_score.shape[1])
            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()

            prob_coarse = torch.empty((cls_score.shape[0], 7), device=cls_score.device)  # [N, 7]
            prob_fine = F.softmax(cls_score, dim=1)  # [N, 52]
            prob_coarse[:, 0] = torch.sum(prob_fine[:, :5], dim=1)
            prob_coarse[:, 1] = torch.sum(prob_fine[:, 5:11], dim=1)
            prob_coarse[:, 2] = torch.sum(prob_fine[:, 11:24], dim=1)
            prob_coarse[:, 3] = torch.sum(prob_fine[:, 24:32], dim=1)
            prob_coarse[:, 4] = torch.sum(prob_fine[:, 32:38], dim=1)
            prob_coarse[:, 5] = torch.sum(prob_fine[:, 38:48], dim=1)
            prob_coarse[:, 6] = torch.sum(prob_fine[:, 48:], dim=1)
            
            gt_coarse = torch.zeros((cls_score.shape[0],1), device=cls_score.device, dtype=torch.int64)  # [N, ]
            for i in range(cls_score.shape[0]):
                # print(label[i])
                gt_coarse[i,0] = fine2coarse(label2[i])

            prob = prob_coarse.gather(1, gt_coarse).view(-1)
            loss_coarse = -self.alpha * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob)
            loss_cls += torch.mean(loss_coarse)

        return loss_cls


@MODELS.register_module()
class BiCrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probability distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
                raise NotImplementedError
            else:
                loss_cls = loss_cls.mean()

            prob_fine = F.softmax(cls_score, dim=1)  # [N, 52]
            prob_coarse_0 = torch.sum(prob_fine[:, :5], dim=1)
            prob_coarse_1 = torch.sum(prob_fine[:, 5:11], dim=1)
            prob_coarse_2 = torch.sum(prob_fine[:, 11:24], dim=1)
            prob_coarse_3 = torch.sum(prob_fine[:, 24:32], dim=1)
            prob_coarse_4 = torch.sum(prob_fine[:, 32:38], dim=1)
            prob_coarse_5 = torch.sum(prob_fine[:, 38:48], dim=1)
            prob_coarse_6 = torch.sum(prob_fine[:, 48:], dim=1)
            
            gt_coarse_0 = torch.sum(label[:, :5], dim=1)
            gt_coarse_1 = torch.sum(label[:, 5:11], dim=1)
            gt_coarse_2 = torch.sum(label[:, 11:24], dim=1)
            gt_coarse_3 = torch.sum(label[:, 24:32], dim=1)
            gt_coarse_4 = torch.sum(label[:, 32:38], dim=1)
            gt_coarse_5 = torch.sum(label[:, 38:48], dim=1)
            gt_coarse_6 = torch.sum(label[:, 48:], dim=1)
            # gt_coarse = torch.argmax(gt_coarse, dim=1, keepdim=True)  # [N, 7]->[N, 1]

            loss_coarse = -(prob_coarse_0 * torch.log(gt_coarse_0)+
                            prob_coarse_1 * torch.log(gt_coarse_1)+
                            prob_coarse_2 * torch.log(gt_coarse_2)+
                            prob_coarse_3 * torch.log(gt_coarse_3)+
                            prob_coarse_4 * torch.log(gt_coarse_4)+
                            prob_coarse_5 * torch.log(gt_coarse_5)+
                            prob_coarse_6 * torch.log(gt_coarse_6))
            loss_cls += 0.5*loss_coarse.mean()

        else:
            # calculate loss for hard label
            raise NotImplementedError
            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls

def fine2coarse(x):
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6