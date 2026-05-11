"""Model factory functions for Cascade Mask R-CNN and HTC.

All return mmdet-compatible config dicts to be passed to MODELS.build().
No classes — each backbone / head is a function.
"""
import os
from typing import Any, Dict

NUM_CLASSES = 4


# ──────────────────────────────────────────────────────────────────────────
#  Backbone factories
# ──────────────────────────────────────────────────────────────────────────

def backbone_resnext101() -> Dict[str, Any]:
    return dict(
        backbone=dict(
            type="ResNeXt", depth=101, groups=64, base_width=4,
            num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1,
            with_cp=True, norm_cfg=dict(type="BN", requires_grad=True),
            style="pytorch",
            init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnext101_64x4d"),
        ),
        neck=fpn_neck([256, 512, 1024, 2048]),
    )


_SWIN_B_CKPT = (
    "/home/longpm/.cache/torch/hub/checkpoints/swin_base_patch4_window7_224_22k_backbone.pth"
    if os.path.exists("/home/longpm/.cache/torch/hub/checkpoints/swin_base_patch4_window7_224_22k_backbone.pth") else
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"
)
_SWIN_L_CKPT = (
    "/home/longpm/.cache/torch/hub/checkpoints/swin_large_patch4_window7_224_22k_backbone.pth"
    if os.path.exists("/home/longpm/.cache/torch/hub/checkpoints/swin_large_patch4_window7_224_22k_backbone.pth") else
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
)


def backbone_swin_b() -> Dict[str, Any]:
    return dict(
        backbone=dict(
            type="SwinTransformer", embed_dims=128, depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32], window_size=7, mlp_ratio=4,
            qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
            drop_path_rate=0.3, patch_norm=True, out_indices=(0, 1, 2, 3),
            with_cp=True,
            init_cfg=dict(type="Pretrained", checkpoint=_SWIN_B_CKPT),
        ),
        neck=fpn_neck([128, 256, 512, 1024]),
    )


def backbone_swin_l() -> Dict[str, Any]:
    return dict(
        backbone=dict(
            type="SwinTransformer", pretrain_img_size=224, embed_dims=192,
            depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=7,
            mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.3, patch_norm=True,
            out_indices=(0, 1, 2, 3), with_cp=True,
            init_cfg=dict(type="Pretrained", checkpoint=_SWIN_L_CKPT),
        ),
        neck=fpn_neck([192, 384, 768, 1536]),
    )


def backbone_convnext_l() -> Dict[str, Any]:
    return dict(
        backbone=dict(
            type="mmpretrain.ConvNeXt", arch="large", out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4, layer_scale_init_value=1.0,
            gap_before_final_norm=False, with_cp=True,
            init_cfg=dict(
                type="Pretrained",
                checkpoint="https://download.openmmlab.com/mmclassification/v0/convnext/"
                           "convnext-large_3rdparty_in21k_20220124-41b5a79f.pth",
                prefix="backbone.",
            ),
        ),
        neck=dict(type="FPN", in_channels=[192, 384, 768, 1536], out_channels=256, num_outs=5),
    )


_BACKBONE_BUILDERS = {
    "resnext101": backbone_resnext101,
    "swin_b": backbone_swin_b,
    "swin_l": backbone_swin_l,
    "convnext_l": backbone_convnext_l,
}


# ──────────────────────────────────────────────────────────────────────────
#  Common heads (FPN, RPN, cascade boxes/masks/test cfg)
# ──────────────────────────────────────────────────────────────────────────

def fpn_neck(in_channels):
    return dict(type="FPN", in_channels=in_channels, out_channels=256, num_outs=5)


def rpn_head():
    return dict(
        type="RPNHead", in_channels=256, feat_channels=256,
        anchor_generator=dict(type="AnchorGenerator", scales=[8],
                              ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type="DeltaXYWHBBoxCoder",
                        target_means=[0, 0, 0, 0], target_stds=[1, 1, 1, 1]),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
    )


def cascade_bbox_stages():
    return [
        dict(
            assigner=dict(type="MaxIoUAssigner", pos_iou_thr=t, neg_iou_thr=t,
                          min_pos_iou=t, match_low_quality=False, ignore_iof_thr=-1),
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.25,
                         neg_pos_ub=-1, add_gt_as_proposals=True),
            mask_size=28, pos_weight=-1, debug=False,
        )
        for t in (0.5, 0.6, 0.7)
    ]


def cascade_test_cfg():
    return dict(
        rpn=dict(nms_pre=1000, max_per_img=1000,
                 nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5),
                  max_per_img=100, mask_thr_binary=0.5),
    )


def _cascade_bbox_heads(num_classes: int):
    return [
        dict(type="Shared2FCBBoxHead", in_channels=256, fc_out_channels=1024,
             roi_feat_size=7, num_classes=num_classes,
             bbox_coder=dict(type="DeltaXYWHBBoxCoder",
                             target_means=[0, 0, 0, 0],
                             target_stds=[s, s, s * 2, s * 2]),
             reg_class_agnostic=True,
             loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
             loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0))
        for s in (0.1, 0.05, 0.033)
    ]


# ──────────────────────────────────────────────────────────────────────────
#  Cascade Mask R-CNN
# ──────────────────────────────────────────────────────────────────────────

def cascade_roi_head(num_classes: int):
    return dict(
        type="CascadeRoIHead", num_stages=3, stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(type="SingleRoIExtractor",
                                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                                out_channels=256, featmap_strides=[4, 8, 16, 32]),
        bbox_head=_cascade_bbox_heads(num_classes),
        mask_roi_extractor=dict(type="SingleRoIExtractor",
                                roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
                                out_channels=256, featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(type="FCNMaskHead", num_convs=4, in_channels=256,
                       conv_out_channels=256, num_classes=num_classes,
                       loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0)),
        train_cfg=dict(rcnn=cascade_bbox_stages()),
        test_cfg=cascade_test_cfg()["rcnn"],
    )


def build_cascade_mask_rcnn(cfg: dict) -> dict:
    backbone_name = cfg.get("backbone", "resnext101").lower()
    num_classes = cfg.get("num_classes", NUM_CLASSES)
    if backbone_name not in _BACKBONE_BUILDERS:
        raise ValueError(f"Unknown backbone '{backbone_name}'. "
                         f"Choices: {list(_BACKBONE_BUILDERS.keys())}")

    bb = _BACKBONE_BUILDERS[backbone_name]()
    return dict(
        type="CascadeRCNN", **bb,
        rpn_head=rpn_head(),
        roi_head=cascade_roi_head(num_classes),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3,
                              min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=0.5,
                             neg_pos_ub=-1, add_gt_as_proposals=False),
                allowed_border=0, pos_weight=-1, debug=False),
            rpn_proposal=dict(nms_pre=2000, max_per_img=2000,
                              nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
            rcnn=cascade_bbox_stages(),
        ),
        test_cfg=cascade_test_cfg(),
    )


# ──────────────────────────────────────────────────────────────────────────
#  HTC (Hybrid Task Cascade)
# ──────────────────────────────────────────────────────────────────────────

def htc_roi_head(num_classes: int):
    bbox_head = _cascade_bbox_heads(num_classes)

    mask_head = [
        dict(type="HTCMaskHead", with_conv_res=False, num_convs=4,
             in_channels=256, conv_out_channels=256, num_classes=num_classes,
             loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0)),
        dict(type="HTCMaskHead", with_conv_res=True, num_convs=4,
             in_channels=256, conv_out_channels=256, num_classes=num_classes,
             loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0)),
        dict(type="HTCMaskHead", with_conv_res=True, num_convs=4,
             in_channels=256, conv_out_channels=256, num_classes=num_classes,
             loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0)),
    ]

    semantic_head = dict(
        type="FusedSemanticHead", num_ins=5, fusion_level=1, num_convs=4,
        in_channels=256, conv_out_channels=256, num_classes=num_classes + 1,
        ignore_label=255,
        loss_seg=dict(type="CrossEntropyLoss", ignore_index=255, loss_weight=0.2),
    )

    return dict(
        type="HybridTaskCascadeRoIHead", interleaved=True, mask_info_flow=True,
        num_stages=3, stage_loss_weights=[1, 0.5, 0.25],
        semantic_roi_extractor=dict(type="SingleRoIExtractor",
                                    roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
                                    out_channels=256, featmap_strides=[8]),
        semantic_head=semantic_head,
        bbox_roi_extractor=dict(type="SingleRoIExtractor",
                                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                                out_channels=256, featmap_strides=[4, 8, 16, 32]),
        bbox_head=bbox_head,
        mask_roi_extractor=dict(type="SingleRoIExtractor",
                                roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
                                out_channels=256, featmap_strides=[4, 8, 16, 32]),
        mask_head=mask_head,
        train_cfg=dict(rcnn=cascade_bbox_stages()),
        test_cfg=cascade_test_cfg()["rcnn"],
    )


def build_htc(cfg: dict) -> dict:
    backbone_name = cfg.get("backbone", "resnext101").lower()
    num_classes = cfg.get("num_classes", NUM_CLASSES)
    if backbone_name not in _BACKBONE_BUILDERS:
        raise ValueError(f"Unknown backbone '{backbone_name}'. "
                         f"Choices: {list(_BACKBONE_BUILDERS.keys())}")

    bb = _BACKBONE_BUILDERS[backbone_name]()
    return dict(
        type="HybridTaskCascade", **bb,
        rpn_head=rpn_head(),
        roi_head=htc_roi_head(num_classes),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3,
                              min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5,
                             neg_pos_ub=-1, add_gt_as_proposals=False),
                allowed_border=0, pos_weight=-1, debug=False),
            rpn_proposal=dict(nms_pre=1000, max_per_img=1000,
                              nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
            rcnn=cascade_bbox_stages(),
        ),
        test_cfg=cascade_test_cfg(),
    )


# ──────────────────────────────────────────────────────────────────────────
#  Top-level dispatcher + mmdet-build helper
# ──────────────────────────────────────────────────────────────────────────

def build_model_cfg(model_cfg: dict) -> dict:
    """Dispatch to the right factory based on model.name."""
    name = model_cfg.get("name", "").lower()
    if name == "cascade_mask_rcnn":
        return build_cascade_mask_rcnn(model_cfg)
    if name == "htc":
        return build_htc(model_cfg)
    raise ValueError(f"Unknown model: {name}")


def build_mmdet_model(model_cfg: dict, device: str):
    """Convert a config dict into an instantiated, weight-initialised mmdet model on `device`."""
    try:
        from mmdet.utils import register_all_modules
        from mmdet.registry import MODELS
        from mmengine.config import ConfigDict
    except ImportError as e:
        raise ImportError("Install mmdet/mmengine/mmcv first. " + str(e)) from e

    register_all_modules()
    try:
        import mmpretrain.models  # noqa: F401  (registers ConvNeXt etc.)
    except ImportError:
        pass
    model = MODELS.build(ConfigDict(model_cfg))
    model.init_weights()
    return model.to(device)
