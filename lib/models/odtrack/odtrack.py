# lib/models/odtrack/odtrack.py

import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.odtrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.odtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
# ---- START OF MODIFICATION ----
from ..lora_util import inject_trainable_moe_kronecker_new
# ---- END OF MODIFICATION ----


class ODTrack(nn.Module):
    """ This is the base class for MMTrack """

    # ---- START OF MODIFICATION ----
    def __init__(self, transformer, box_head, cfg, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            cfg: The experiment configuration dictionary.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.cfg = cfg  # Store cfg

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len
        # This will be populated by build_odtrack if MoE is used
        self.moe_params = None
    # ---- END OF MODIFICATION ----

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        for i in range(len(search)):
            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, track_query=self.track_query, token_len=self.token_len)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach() # stop grad  (B, N, C)
                
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            
            # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['backbone_feat'] = x
            
            # ---- START OF MODIFICATION ----
            # Collect MoE auxiliary loss from the backbone
            if self.cfg.MODEL.BACKBONE.USE_MOE:
                aux_loss_total = 0.0
                for module in self.backbone.modules():
                    if hasattr(module, 'aux_loss'):
                        aux_loss_total += module.aux_loss
                out['aux_loss'] = aux_loss_total
            # ---- END OF MODIFICATION ----
            
            out_dict.append(out)
            
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_odtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )

    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = ODTrack(
        backbone,
        box_head,
        # ---- START OF MODIFICATION ----
        cfg,
        # ---- END OF MODIFICATION ----
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )

    # ---- START OF MODIFICATION ----
    # Inject MoE layers if configured, regardless of training or testing mode
    if cfg.MODEL.BACKBONE.USE_MOE:
        print("Injecting Mixture of Experts (MoE) layers...")
        
        for param in model.backbone.parameters():
            param.requires_grad = False
            
        # Select the correct injection function based on moe_type
        if cfg.MODEL.BACKBONE.MOE_TYPE == 'kronecker':
            injection_fn = inject_trainable_moe_kronecker_new
        elif cfg.MODEL.BACKBONE.MOE_TYPE == 'lora':
            # Assuming MOELoraInjectedLinear is the desired class from lora_util.py
            injection_fn = inject_trainable_moe_lora
        else:
            raise ValueError(f"Unknown MoE type: {cfg.MODEL.BACKBONE.MOE_TYPE}")

        moe_params_generators, _ = injection_fn(
            model.backbone,
            target_replace_module=set(cfg.MODEL.BACKBONE.MOE_TARGET_MODULES),
            where=cfg.MODEL.BACKBONE.MOE_WHERE,
            # Pass new parameters to the injection function
            n=cfg.MODEL.BACKBONE.MOE_EXPERTS,
            ranks=cfg.MODEL.BACKBONE.MOE_RANKS,
            top_k=cfg.MODEL.BACKBONE.MOE_TOP_K
        )

        if training:
            moe_params_list = []
            for param_gen in moe_params_generators:
                if isinstance(param_gen, torch.nn.Parameter):
                     moe_params_list.append(param_gen)
                else:
                     moe_params_list.extend(list(param_gen))
            
            model.moe_params = moe_params_list
            print(f"Successfully injected MoE layers for training.")
        else:
            print("Successfully injected MoE layers for testing.")
    # ---- END OF MODIFICATION ----

    return model