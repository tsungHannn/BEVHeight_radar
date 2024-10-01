from torch import nn

from layers.backbones.lss_fpn import LSSFPN
from layers.heads.bev_height_head import BEVHeightHead
# Radar branch
from layers.backbones.pts_backbone import PtsBackbone
# Fusion module
from layers.fusion.fusion_module import RCFuser

__all__ = ['BEVHeight']


class BEVHeight(nn.Module):
    """
    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_height (bool): Whether to return height.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, backbone_pts_conf, is_train_height=False):
        super(BEVHeight, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)


        # Radar branch
        self.backbone_pts = PtsBackbone(**backbone_pts_conf)

        # Fusion module
        self.fuser = RCFuser(in_channels=80) # to do

        self.head = BEVHeightHead(**head_conf)
        self.is_train_height = is_train_height

    def forward(
        self,
        x,
        mats_dict,
        sweep_ptss=None,
        timestamps=None,
    ):
        """Forward function for BEVHeight

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            sweep_ptss (Tensor): Input points.
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        # print("sweep_ptss.shape:", sweep_ptss.shape) #sweep_ptss.shape: torch.Size([2, 1, 1, 1000, 6])
        if self.is_train_height and self.training:
            x, height_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_height=True)
            preds = self.head(x)
            return preds, height_pred
        else:
            x = self.backbone(x, mats_dict, timestamps)
            # Radar branch
            ptss_context, ptss_occupancy, _ = self.backbone_pts(sweep_ptss) 
            # print("ptss_context.shape", ptss_context.shape) # ptss_context.shape torch.Size([2, 1, 80, 70, 44])
            # Fusion module
            fused = self.fuser(x, ptss_context)
            preds = self.head(fused) # 用合併後的feature進行預測
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)


    def loss(self, targets, preds_dicts):
        """Loss function for BEVHeight.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        loss = self.head.loss(targets, preds_dicts)
        # print('model.loss')
        # print(loss)
        return loss

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
