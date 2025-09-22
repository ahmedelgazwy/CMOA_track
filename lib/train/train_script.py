import os
# loss function related
from lib.utils.box_ops import giou_loss, iouhead_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss

# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.odtrack import build_odtrack
# forward propagation related
from lib.train.actors import ODTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    # ---- START OF MODIFICATION ----
    # Override cfg with command-line arguments if provided
    if settings.moe_experts is not None:
        cfg.MODEL.BACKBONE.MOE_EXPERTS = settings.moe_experts
    if settings.moe_ranks is not None:
        cfg.MODEL.BACKBONE.MOE_RANKS = settings.moe_ranks
    if settings.moe_top_k is not None:
        cfg.MODEL.BACKBONE.MOE_TOP_K = settings.moe_top_k
    if settings.moe_where is not None:
        cfg.MODEL.BACKBONE.MOE_WHERE = settings.moe_where
    if settings.moe_type is not None:
        cfg.MODEL.BACKBONE.MOE_TYPE = settings.moe_type
    # ---- END OF MODIFICATION ----
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "odtrack":
        # Pass training=True to ensure MoE layers are injected
        net = build_odtrack(cfg, training=True)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
        
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    
    # Loss functions and Actors
    if settings.script_name == "odtrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.0, 'cls': 1.0}
        actor = ODTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # ---- START OF MODIFICATION ----
    model_without_ddp = net.module if settings.local_rank != -1 else net
     
    if cfg.MODEL.BACKBONE.USE_MOE:
        # If MoE is used, we only want to train the head and the new adapter parameters.
        # The original backbone parameters were already set to requires_grad=False in `build_odtrack`.
        print("MoE training is enabled. The optimizer will only see head and adapter parameters.")
        # The get_optimizer_scheduler function will now automatically pick up only the parameters
        # with requires_grad=True, which are the head and the MoE adapters.
        # We just need to make sure the learning rate for the backbone (adapters) is not scaled down.
        cfg.TRAIN.BACKBONE_MULTIPLIER = 1.0 
    # ---- END OF MODIFICATION ----

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)