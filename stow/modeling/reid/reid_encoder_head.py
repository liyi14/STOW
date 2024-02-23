import torch 
from torch import nn
import torch.nn.functional as F
from stow.modeling.transformer_decoder.frame_mask2former_transformer_decoder import MLP

class reid_encoder_head(nn.Module):
    def __init__(self, hidden_dim, reid_dim, reid_cfg, seg_seg_head_cfg):
        super().__init__()
        encoder_type = reid_cfg.ENCODER_TYPE
        pos_embed_type = reid_cfg.ENCODER_POS_TYPE
        reid_normalize = reid_cfg.REID_NORMALIZE
        num_encoder_layers = reid_cfg.ENCODER_NUM_LAYERS
        encoder_layer_structure = reid_cfg.ENCODER_LAYER_STRUCTURE
        encoder_activation = reid_cfg.ENCODER_ACTIVATION
        encoder_dim_feedforward = reid_cfg.ENCODER_DIM_FEEDFORWARD
        normalize_before = reid_cfg.ENCODER_NORMALIZE_BEFORE
        last_layer_bias = reid_cfg.ENCODER_LAST_LAYER_BIAS
        self.encoder_type = encoder_type
        if encoder_type == "linear":
            self.reid_head = nn.Linear(hidden_dim, reid_dim, bias=last_layer_bias)
        elif encoder_type == "mlp":
            activation = nn.ReLU() if encoder_activation == "relu" else nn.GELU()
            self.reid_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                # nn.ReLU(),
                activation,
                nn.Linear(hidden_dim, reid_dim, bias=last_layer_bias)
            )
        elif encoder_type == "attention" or encoder_type == "full-attention":
            from .interact_transformer import build_transformer_encoder
            self.reid_head = build_transformer_encoder(d_model=hidden_dim,
                                                       num_encoder_layers=num_encoder_layers,
                                                       activation=encoder_activation,
                                                       layer_structure=encoder_layer_structure,
                                                       dim_feedforward=encoder_dim_feedforward,
                                                       normalize_before=normalize_before)
            if reid_cfg.ENCODER_NORM_BEFORE_HEAD:
                self.encoder_norm = nn.LayerNorm(hidden_dim)
            else:
                self.encoder_norm = nn.Identity()
            if reid_cfg.ENCODER_ONE_LAYER_AFTER_ATTENTION:
                self.layer_after_encoder = nn.Linear(hidden_dim, reid_dim, last_layer_bias)
            else:
                self.layer_after_encoder = nn.Identity()
            if reid_cfg.ENCODER_UPDATE_CONFIDENCE:
                self.cls_head = nn.Linear(hidden_dim, 2)
            else:
                self.cls_head = None
            if reid_cfg.ENCODER_PREDICT_MASK:
                self.mask_head = MLP(hidden_dim, hidden_dim, seg_seg_head_cfg.MASK_DIM, 3)
            else:
                self.mask_head = None
            self.pos_embed = nn.Embedding(100, hidden_dim)
            if pos_embed_type == 'fixed':
                self.pos_embed.weight.requires_grad = False
            self.pos_embed_type = pos_embed_type
            self.pos_embed_inited = False
        else:
            raise NotImplemented
        self.reid_normalize = reid_normalize
        
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
        
        if encoder_type == 'mlp':
            self.reid_head[0].bias.data.zero_()
        if last_layer_bias:
            if encoder_type == 'linear':
                self.reid_head.bias.data.zero_()
            if encoder_type == 'mlp':
                self.reid_head[2].bias.data.zero_()
            elif (encoder_type == 'attention' or encoder_type == 'full-attention') and reid_cfg.ENCODER_ONE_LAYER_AFTER_ATTENTION:
                self.layer_after_encoder.bias.data.zero_()
            else:
                pass
            
    def forward(self, batched_outputs, pos_embed=None):
        if self.encoder_type == 'linear' or self.encoder_type == 'mlp':
            for output in batched_outputs:
                for aux in output['aux_outputs']:
                    pred_reid = self.reid_head(aux['meta_embedding'])
                    if self.reid_normalize:
                        pred_reid = F.normalize(pred_reid, dim=-1)
                    aux['pred_reid'] = pred_reid
                pred_reid = self.reid_head(output['meta_embedding'])
                if self.reid_normalize:
                    pred_reid = F.normalize(pred_reid, dim=-1)
                output['pred_reid'] = pred_reid
        elif self.encoder_type == 'full-attention' or self.encoder_type == 'attention':
            # init pos_embed with external pos_embed
            if self.pos_embed_inited == False or self.pos_embed_type == 'external':
                self.pos_embed.weight.data = pos_embed.clone().detach()
                self.pos_embed_inited = True
            # print("reid pos_embed.weight.data.sum()", self.pos_embed.weight.data.sum())
            # print("decoder pos_embed.sum()", pos_embed.sum())
            var_must_exist = batched_outputs[0]['meta_embedding']
            bs, q, c = var_must_exist.shape
            t = len(batched_outputs)
            # prepare src_pos, QxC -> TxNxQxC -> (TxQ)xNxC
            # assert self.pose_embed.shape == (q, c)
            pos_embed = self.pos_embed.weight[None, None, :, :].expand(t, bs, q, c).permute(0,2,1,3).reshape(t*q, bs, c).to(var_must_exist.device)
            if self.pos_embed_type == 'zero':
                pos_embed = None
            elif self.pos_embed_type == 'fixed' or self.pos_embed_type == 'external':
                pos_embed = pos_embed.detach()
            elif self.pos_embed_type == 'learned':
                pass
            else:
                raise KeyError
            for aux_idx in range(len(batched_outputs[0]['aux_outputs'])):
                # prepare src_feat, Tx[NxQxC] -> TxNxQxC -> (TxQ)xNxC
                src = torch.stack([output['aux_outputs'][aux_idx]['meta_embedding'] for output in batched_outputs], dim=0)
                src = src.permute(0,2,1,3).reshape(t*q, bs, c)
                
                encoder_output = self.reid_head(src, pos_embed=pos_embed)
                encoder_output = self.encoder_norm(encoder_output)
                pred_reid = self.layer_after_encoder(encoder_output).permute(1,0,2)
                for frame_idx in range(t):
                    batched_outputs[frame_idx]['aux_outputs'][aux_idx]['pred_reid'] = pred_reid[:, frame_idx*q:(frame_idx+1)*q]
                    
                if self.cls_head is not None:
                    pred_cls = self.cls_head(encoder_output).permute(1,0,2)
                    for frame_idx in range(t):
                        batched_outputs[frame_idx]['aux_outputs'][aux_idx]['pred_cls'] = pred_cls[:, frame_idx*q:(frame_idx+1)*q]
                        
                if self.mask_head is not None:
                    mask_embed = self.mask_head(encoder_output).permute(1,0,2)
                    for frame_idx in range(t):
                        pred_masks = torch.einsum("bqc,bchw->bqhw", 
                                                  mask_embed[:, frame_idx*q:(frame_idx+1)*q], 
                                                  batched_outputs[frame_idx]['mask_feature'])
                        batched_outputs[frame_idx]['aux_outputs'][aux_idx]['pred_masks'] = pred_masks
            src = torch.stack([output['meta_embedding'] for output in batched_outputs], dim=0)
            src = src.permute(0,2,1,3).reshape(t*q, bs, c)
            encoder_output = self.reid_head(src, pos_embed=pos_embed)
            pred_reid = self.layer_after_encoder(encoder_output).permute(1,0,2)
            for frame_idx in range(t):
                batched_outputs[frame_idx]['pred_reid'] = pred_reid[:, frame_idx*q:(frame_idx+1)*q]
            if self.cls_head is not None:
                pred_cls = self.cls_head(encoder_output).permute(1,0,2)
                for frame_idx in range(t):
                    batched_outputs[frame_idx]['pred_cls'] = pred_cls[:, frame_idx*q:(frame_idx+1)*q]
            if self.mask_head is not None:
                mask_embed = self.mask_head(encoder_output).permute(1,0,2)
                for frame_idx in range(t):
                    pred_masks = torch.einsum("bqc,bchw->bqhw", 
                                                mask_embed[:, frame_idx*q:(frame_idx+1)*q], 
                                                batched_outputs[frame_idx]['mask_feature'])
                    batched_outputs[frame_idx]['pred_masks'] = pred_masks
        else:
            raise KeyError