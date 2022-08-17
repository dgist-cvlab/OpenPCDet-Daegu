import torch.nn as nn
# import torch

from .pointpillar import PointPillar
from .detector3d_template import Detector3DTemplate
from ...utils.ema_pytorch import EMA

import copy

class PointPillarDual(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        # super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # self.module_list = self.build_networks()
        super().__init__(model_cfg, num_class, dataset)
        self.doEMA = True
        self.model_direct = PointPillar(model_cfg, num_class, dataset)
        self.model_shadow = PointPillar(model_cfg, num_class, dataset)
        self.model_shadow.requires_grad_(False)
        self.ema = EMA(
                self.model_direct,
                ema_model = self.model_shadow,
                beta = 0.9999,              # exponential moving average factor
                update_after_step = 100,    # only after this number of .update() calls will it start updating
                update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
            )
        self.add_module(
            'mseloss',
            nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        )
    
    def update_ema(self):
        self.ema.update()

    def forward(self, batch_dict):
        if self.training:
            batch_shadow = copy.deepcopy(batch_dict)
            batch_shadow['points'] = batch_shadow.pop('points_shadow')
            batch_shadow['gt_boxes'] = batch_shadow.pop('gt_boxes_shadow')

        for cur_module in self.model_direct.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            for cur_module in self.model_shadow.module_list:
                batch_shadow = cur_module(batch_shadow)

            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict, batch_shadow)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.model_direct.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict, batch_shadow):
        disp_dict = {}

        # default loss
        loss_rpn, tb_dict = self.model_direct.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        # consistency loss
        loss_consist = self.mseloss(batch_dict['spatial_features_2d'], batch_shadow['spatial_features_2d'])

        loss = loss_rpn + loss_consist
        return loss, tb_dict, disp_dict
