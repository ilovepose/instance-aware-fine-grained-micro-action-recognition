# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import normal_init

from mmaction.registry import MODELS
from mmaction.evaluation import top_k_accuracy
from .base import AvgConsensus, BaseHead


@MODELS.register_module()
class MANet3DHeadV0(BaseHead):
    """Class head for MANet3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        num_segments (int): Number of frame segments. Default: 8.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_emb=dict(type='MseLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.8,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.loss_emb = MODELS.build(loss_emb)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        
        self.fc_emb = nn.Linear(self.in_channels, 300)
        self.fc_emb_t=nn.Linear(300,300)
        self.tanh=nn.Tanh()

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.fc_emb, std=self.init_std)
        normal_init(self.fc_emb_t, std=self.init_std)


    def forward(self, x, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Useless in TSMHead. By default, `num_segs`
                is equal to `clip_len * num_clips * num_crops`, which is
                automatically generated in Recognizer forward phase and
                useless in TSM models. The `self.num_segments` we need is a
                hyper parameter to build TSM models.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if self.avg_pool is not None:
            x = self.avg_pool(x)  # N,C,H,W->N,C,1,1
        
        if self.dropout is not None:
            x = self.dropout(x)

        x = torch.flatten(x, 1)  #  N,C,1,1->N,C
        cls_score = self.fc_cls(x)  # N,C,H,W
        emb_score = self.fc_emb_t(self.tanh(self.fc_emb(x)))
        return cls_score, emb_score  # [N,Cls]

    def loss(self, feats, data_samples, **kwargs):
        cls_scores, emb_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples, emb_scores)
    
    def loss_by_feat(self, cls_scores, data_samples, emb_scores):
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        emb_gt = [data_sample.gt_emb for data_sample in data_samples]  # TODO: Move to preprocessing
        emb_gt = torch.stack(emb_gt, dim=0)
        loss_embd = self.loss_emb(emb_scores, emb_gt, labels)*50
        loss_cls += loss_embd
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict(self, feats, data_samples, **kwargs):
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores, _ = self(feats, **kwargs)
        # cls_scores = self.average_clip(cls_scores)  # TODO: align to baseline, self.num_segments
        return self.predict_by_feat(cls_scores, data_samples)

@MODELS.register_module()
class MANet3DHeadV1(BaseHead):
    """Class head for MANet3D-V1.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        num_segments (int): Number of frame segments. Default: 8.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_emb=dict(type='MseLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.001,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.loss_emb = MODELS.build(loss_emb)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        self.fc_emb = nn.Linear(self.in_channels, 250)
        self.fc_emb_t=nn.Linear(250,250)
        self.tanh=nn.Tanh()

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.fc_emb, std=self.init_std)
        normal_init(self.fc_emb_t, std=self.init_std)

    def forward(self, x, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Useless in TSMHead. By default, `num_segs`
                is equal to `clip_len * num_clips * num_crops`, which is
                automatically generated in Recognizer forward phase and
                useless in TSM models. The `self.num_segments` we need is a
                hyper parameter to build TSM models.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if self.avg_pool is not None:
            x = self.avg_pool(x)  # N,C,4,7,7->N,C,1,1,1
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = torch.flatten(x, 1)  #  N,C,1,1->N,C
        # [N, in_channels]
        cls_score = self.fc_cls(x)  # N,C
        emb_score = self.fc_emb_t(self.tanh(self.fc_emb(x)))
        return cls_score, emb_score  # [N,Cls]

    def loss(self, feats, data_samples, **kwargs):
        cls_scores, emb_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples, emb_scores)

    def loss_by_feat(self, cls_scores, data_samples, emb_scores):
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        emb_gt = [data_sample.gt_emb for data_sample in data_samples]  # TODO: Move to preprocessing
        emb_gt = torch.stack(emb_gt, dim=0).to(emb_scores.device)
        loss_embd = self.loss_emb(emb_scores, emb_gt, labels)*50
        loss_cls += loss_embd
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict(self, feats, data_samples, **kwargs):
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores, _ = self(feats, **kwargs)
        # cls_scores = self.average_clip(cls_scores)  # TODO: align to baseline, self.num_segments
        return self.predict_by_feat(cls_scores, data_samples)
    
@MODELS.register_module()
class MANet3DHead(BaseHead):
    """Classification head for MANet3DHeadV2.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        num_coarse = 7  #MultiFocalLoss, num_class=num_coarse, alpha=0.25
        self.loss_coarse = MODELS.build(dict(type='CrossEntropyLoss'))
        self.fc_coarse = nn.Linear(self.in_channels, num_coarse)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.fc_coarse, std=self.init_std)

    def forward(self, x, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        cls_coarse= self.fc_coarse(x)
        # [N, num_classes], [N, num_coarse]
        return cls_score, cls_coarse

    def loss(self, feats, data_samples, **kwargs):
        cls_scores, cls_coarses = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples, cls_coarses)

    def loss_by_feat(self, cls_scores, data_samples, cls_coarses):
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        # cls_coarses = F.softmax(cls_coarses, dim=1)  # P(A)->P'(A)
        gt_coarses  = torch.empty_like(cls_coarses)  # [N, 7]
        gt_coarses[:, 0] = torch.sum(labels[:, :5], dim=1)
        gt_coarses[:, 1] = torch.sum(labels[:, 5:11], dim=1)
        gt_coarses[:, 2] = torch.sum(labels[:, 11:24], dim=1)
        gt_coarses[:, 3] = torch.sum(labels[:, 24:32], dim=1)
        gt_coarses[:, 4] = torch.sum(labels[:, 32:38], dim=1)
        gt_coarses[:, 5] = torch.sum(labels[:, 38:48], dim=1)
        gt_coarses[:, 6] = torch.sum(labels[:, 48:], dim=1)
        loss_coarse = self.loss_coarse(cls_coarses, gt_coarses)

        # cls_coarses_repeat = torch.empty_like(cls_scores)  # [N, 7]->[N, 52]
        # cls_coarses_repeat[:, :5]    = cls_coarses[:, [0]]
        # cls_coarses_repeat[:, 5:11]  = cls_coarses[:, [1]]
        # cls_coarses_repeat[:, 11:24] = cls_coarses[:, [2]]
        # cls_coarses_repeat[:, 24:32] = cls_coarses[:, [3]]
        # cls_coarses_repeat[:, 32:38] = cls_coarses[:, [4]]
        # cls_coarses_repeat[:, 38:48] = cls_coarses[:, [5]]
        # cls_coarses_repeat[:, 48:]   = cls_coarses[:, [6]]
        # cls_scores = torch.sigmoid(cls_scores)  # P(B|A)->P'(B|A) and 0<value<1
        # cls_scores = cls_scores * cls_coarses_repeat.detach()  # P'(B|A)*P'(A)->P'(B)
        
        loss_cls = self.loss_cls(cls_scores, labels)
        loss_cls += loss_coarse

        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict(self, feats, data_samples, **kwargs):
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores, cls_coarses = self(feats, **kwargs)
        cls_coarses = F.softmax(cls_coarses, dim=1)
        cls_coarses_repeat = torch.empty_like(cls_scores)  # [N, 7]->[N, 52]
        cls_coarses_repeat[:, :5]    = cls_coarses[:, [0]]
        cls_coarses_repeat[:, 5:11]  = cls_coarses[:, [1]]
        cls_coarses_repeat[:, 11:24] = cls_coarses[:, [2]]
        cls_coarses_repeat[:, 24:32] = cls_coarses[:, [3]]
        cls_coarses_repeat[:, 32:38] = cls_coarses[:, [4]]
        cls_coarses_repeat[:, 38:48] = cls_coarses[:, [5]]
        cls_coarses_repeat[:, 48:]   = cls_coarses[:, [6]]
        cls_scores = torch.sigmoid(cls_scores)
        cls_scores = cls_scores * cls_coarses_repeat.detach()
        # cls_scores = self.average_clip(cls_scores)  # TODO: align to baseline, self.num_segments
        return self.predict_by_feat(cls_scores, data_samples)


@MODELS.register_module()
class MANet3DHeadTemp(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type = 'avg',
                 dropout_ratio = 0.5,
                 init_std = 0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.fc_coarse = nn.Linear(self.in_channels, 7)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        cls_coarse = self.fc_coarse(x)
        # [N, 7]
        return cls_score