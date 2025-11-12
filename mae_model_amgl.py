import os
import sys
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import SAGEConv,LayerNorm
from mae_utils import get_sinusoid_encoding_table,Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torch_geometric.nn import SAGEConv,GCNConv,SGConv,GATConv,TransformerConv,LayerNorm,SAGPooling,ClusterGCNConv,JumpingKnowledge
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DynamicJKGCN(nn.Module):
    def __init__(self, in_feats, out_classes, num_layers=3, mode='dynamic'):
        super(DynamicJKGCN, self).__init__()

        self.mode = mode
        self.num_layers = num_layers
        self.dropout_ratio = 0.05

        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.convs.append(GCNConv(in_feats, out_classes * 2))
        self.dropouts.append(nn.Dropout(p=self.dropout_ratio))

        for _ in range(1, num_layers):
            self.convs.append(GCNConv(out_classes * 2, out_classes * 2))
            self.dropouts.append(nn.Dropout(p=self.dropout_ratio))

        # 动态Jumping Knowledge
        if mode == 'dynamic':
            self.jk_weights = nn.Parameter(torch.randn(num_layers))

        self.fc = nn.Linear(out_classes * 2, out_classes)

        self.historical_activations = None
        self.edge_index = None

    def forward(self, x, edge_index, neighbor_sampling=True, alpha=0.3):
        self.edge_index = edge_index
        layer_out = []

        for layer in range(self.num_layers):
            x = self.convs[layer](x, edge_index)
            x = F.relu(x)
            x = self.dropouts[layer](x)

            if layer == 0 and neighbor_sampling:
                x = self.sample_neighbors(x)

            # 控制变量
            if layer == 1:
                if self.historical_activations is None or self.historical_activations.size(0) != x.size(0):
                    self.historical_activations = x.clone().detach()

                delta_x = x - self.historical_activations
                x = self.historical_activations + alpha * delta_x
                self.historical_activations = x.clone().detach()

            layer_out.append(x)

        if self.mode == 'dynamic':
            norm_weights = F.softmax(self.jk_weights, dim=0)
            h = sum(w * h_i for w, h_i in zip(norm_weights, layer_out))
        else:
            h = layer_out[-1]

        h = self.fc(h)
        return h

    def sample_neighbors(self, activations, min_samples=3, scale_factor=2):

        edge_index = self.edge_index
        num_nodes = activations.size(0)

        adj_dict = {i: [] for i in range(num_nodes)}
        row, col = edge_index
        for src, dst in zip(row.tolist(), col.tolist()):
            adj_dict[src].append(dst)

        sampled_activations = []

        for node in range(num_nodes):
            neighbors = adj_dict[node]
            degree = len(neighbors)

            if degree == 0:
                agg = activations[node]
            else:
                num_samples = max(
                    math.ceil(degree * scale_factor),
                    min_samples
                )

                if degree <= num_samples:
                    sampled_neighbors = neighbors
                else:
                    node_feat = activations[node]
                    sims = []
                    for nei in neighbors:
                        nei_feat = activations[nei]
                        sim = F.cosine_similarity(node_feat.unsqueeze(0), nei_feat.unsqueeze(0))
                        sims.append((sim.item(), nei))
                    sims = sorted(sims, key=lambda x: -x[0])
                    sampled_neighbors = [idx for _, idx in sims[:num_samples]]

                neighbor_feats = activations[sampled_neighbors]
                agg = neighbor_feats.mean(dim=0)

            sampled_activations.append(agg)

        return torch.stack(sampled_activations, dim=0)



def reset(module):
    if module is None:
        return
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
    else:
        for submodule in module.children():
            reset(submodule)

class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)


    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        weighted_x = gate * x
        return out, gate, weighted_x


    
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    
class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False,train_type_num=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

        self.patch_embed = nn.Linear(embed_dim,embed_dim)
        num_patches = train_type_num

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=512, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,train_type_num=3,
                 ):
        super().__init__()
        self.num_classes = num_classes
#         assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=512, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=512, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.3,
                 drop_path_rate=0.3, 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 train_type_num=3,
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            train_type_num=train_type_num)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=3,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            train_type_num=train_type_num)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
#         self.mask_token = torch.zeros(1, 1, decoder_embed_dim).to(device)
        

        self.pos_embed = get_sinusoid_encoding_table(train_type_num, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]

        B, N, C = x_vis.shape
        
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.decoder(x_full, 0) # [B, N_mask, 3 * 16 * 16]

        tmp_x = torch.zeros_like(x).to(device)
        Mask_n = 0
        Truth_n = 0
        for i,flag in enumerate(mask[0][0]):
            if flag:  
                tmp_x[:,i] = x[:,pos_emd_vis.shape[1]+Mask_n]
                Mask_n += 1
            else:
                tmp_x[:,i] = x[:,Truth_n]
                Truth_n += 1
        return tmp_x



def Mix_mlp(dim1):
    
    return nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.GELU(),
            nn.Linear(dim1, dim1))

class MixerBlock(nn.Module):
    def __init__(self,dim1,dim2):
        super(MixerBlock,self).__init__() 
        
        self.norm = LayerNorm(dim1)
        self.mix_mip_1 = Mix_mlp(dim1)
        self.mix_mip_2 = Mix_mlp(dim2)
        
    def forward(self,x): 
        
        y = self.norm(x)
        y = y.transpose(0,1)
        y = self.mix_mip_1(y)
        y = y.transpose(0,1)
        x = x + y
        y = self.norm(x)
        x = x + self.mix_mip_2(y)
        
#         y = self.norm(x)
#         y = y.transpose(0,1)
#         y = self.mix_mip_1(y)
#         y = y.transpose(0,1)
#         x = self.norm(y)
        return x



def MLP_Block(dim1, dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout))

def GNN_relu_Block(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
#             GATConv(in_channels=dim1,out_channels=dim2),
            nn.ReLU(),
            LayerNorm(dim2),
            nn.Dropout(p=dropout))


class SequentialSAGEConv(nn.Module):
    def __init__(self, in_feats, out_classes):
        super(SequentialSAGEConv, self).__init__()
        self.conv1 = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        self.conv2 = SAGEConv(in_channels=out_classes, out_channels=out_classes // 2)
        self.conv3 = SAGEConv(in_channels=out_classes // 2, out_channels=out_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x




from timm.models.layers import trunc_normal_ as __call_trunc_normal_


# 新增一行（放在同一区域）
timm_trunc_normal_ = __call_trunc_normal_
from torch_geometric.utils import softmax as pyg_softmax


class MGFF(nn.Module):
    """
    Multi-modal Graph Feature Fusion (MGFF)
    """
    def __init__(self, dim: int = 512, dropout: float = 0.5, gn_groups: int = 8):
        super().__init__()
        self.dim = dim

        self.lin_FH = nn.Linear(dim, dim, bias=True)
        self.lin_FW = nn.Linear(dim, dim, bias=True)
        self.mix_sp = nn.Linear(2 * dim, dim, bias=True)


        self.dw_scale = nn.Parameter(torch.ones(dim))
        self.dw_bias  = nn.Parameter(torch.zeros(dim))
        self.norm_ch  = nn.GroupNorm(num_groups=gn_groups, num_channels=dim)
        self.conv1_ch = nn.Linear(dim, dim, bias=True)

        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.reset_parameters()

    def reset_parameters(self):
        timm_trunc_normal_(self.lin_FH.weight, std=0.02)
        timm_trunc_normal_(self.lin_FW.weight, std=0.02)
        timm_trunc_normal_(self.mix_sp.weight, std=0.02)
        timm_trunc_normal_(self.conv1_ch.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.ones_(self.dw_scale)
        nn.init.zeros_(self.dw_bias)

    @staticmethod
    def _softmax_channel(v: torch.Tensor) -> torch.Tensor:
        M, C = v.shape
        src = v.reshape(-1)                                      # [M*C]
        idx = torch.arange(M, device=v.device).repeat_interleave(C)
        out = pyg_softmax(src, idx, num_nodes=M)                 # [M*C]
        return out.view(M, C)

    @staticmethod
    def _mean_per_group(v: torch.Tensor) -> torch.Tensor:
        # 组均值 (C-AvgPool 1D 退化)：[M,C] -> [M,1]
        M, C = v.shape
        src = v.reshape(-1)                                      # [M*C]
        idx = torch.arange(M, device=v.device).repeat_interleave(C)
        summed = scatter_add(src, idx, dim=0, dim_size=M)        # [M]
        mean = summed / C
        return mean.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(0) == 3 and x.size(1) == self.dim, \
            f"Expected [3,{self.dim}], got {tuple(x.shape)}"
        M, C = x.shape

        FH = self.lin_FH(x)
        FW = self.lin_FW(x)
        A_sp = self.mix_sp(torch.cat([FH, FW], dim=-1))
        O_sp = A_sp * x


        c_avg = self._mean_per_group(x).expand_as(x)
        dw    = c_avg * self.dw_scale + self.dw_bias
        gn_out = self.norm_ch(dw)
        ch_feat = self.conv1_ch(gn_out)
        O_ch = ch_feat * x

        # ===== Fusion gate =====
        S_soft = self._softmax_channel(O_sp) * self._softmax_channel(O_ch)
        S_avg  = self._mean_per_group(O_sp).expand_as(x) * self._mean_per_group(O_ch).expand_as(x)
        gate = S_soft + S_avg

        O_fused = gate * x
        out = torch.stack([self.head(O_fused[m]) for m in range(M)], dim=0)
        return out





class fusion_model_mae_2(nn.Module):
    def __init__(self,in_feats,n_hidden,out_classes,dropout=0.3,train_type_num=3):
        super(fusion_model_mae_2,self).__init__()
        self.attention = my_GlobalAttention(gate_nn=torch.nn.Linear(out_classes, 512))
        self.img_gnn_2 = DynamicJKGCN(in_feats =in_feats,out_classes=out_classes)
        self.img_relu_2 = GNN_relu_Block(out_classes)  
        self.rna_gnn_2 = GCNConv(in_channels=in_feats,out_channels=out_classes)
        self.rna_relu_2 = GNN_relu_Block(out_classes)      
        self.cli_gnn_2 = GCNConv(in_channels=in_feats,out_channels=out_classes)
        self.cli_relu_2 = GNN_relu_Block(out_classes) 
        # TransformerConv

        att_net_img = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_img = my_GlobalAttention(att_net_img)

        att_net_rna = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_rna = my_GlobalAttention(att_net_rna)        

        att_net_cli = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_cli = my_GlobalAttention(att_net_cli)


        att_net_img_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_img_2 = my_GlobalAttention(att_net_img_2)

        att_net_rna_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_rna_2 = my_GlobalAttention(att_net_rna_2)        

        att_net_cli_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_cli_2 = my_GlobalAttention(att_net_cli_2)

        
        self.mae = PretrainVisionTransformer(encoder_embed_dim=out_classes, decoder_num_classes=out_classes, decoder_embed_dim=out_classes, encoder_depth=1,decoder_depth=1,train_type_num=train_type_num)
        self.mix = MGFF()

        self.lin1_img = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_img = torch.nn.Linear(out_classes//4,1)

        self.lin1_rna = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_rna = torch.nn.Linear(out_classes//4,1) 

        self.lin1_cli = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_cli = torch.nn.Linear(out_classes//4,1)         

        self.norm_img = LayerNorm(out_classes//4)
        self.norm_rna = LayerNorm(out_classes//4)
        self.norm_cli = LayerNorm(out_classes//4)
        self.relu = torch.nn.ReLU() 
        self.dropout=nn.Dropout(p=dropout)


    def forward(self,all_thing,train_use_type=None,use_type=None,in_mask=[],mix=True):

        if len(in_mask) == 0:
            mask = np.array([[[False]*len(train_use_type)]])
        else:
            mask = in_mask

        data_type = use_type
        x_img = all_thing.x_img
        x_rna = all_thing.x_rna
        x_cli = all_thing.x_cli

        data_id=all_thing.data_id
        edge_index_img=all_thing.edge_index_image
        edge_index_rna=all_thing.edge_index_rna
        edge_index_cli=all_thing.edge_index_cli

        
        save_fea = {}
        fea_dict = {}
        num_img = len(x_img)
        num_rna = len(x_rna)
        num_cli = len(x_cli)      
               
            
        att_2 = []
        att_each_2 = []
        pool_x = torch.empty((0)).to(device)
        if 'img' in data_type:
            x_img = self.img_gnn_2(x_img,edge_index_img) 
            x_img = self.img_relu_2(x_img)
            batch = torch.zeros(len(x_img),dtype=torch.long).to(device)
            pool_x_img,att_img_2,att_img_each = self.mpool_img(x_img,batch)
            att_2.append(att_img_2)
            att_each_2.append(att_img_each)
            pool_x = torch.cat((pool_x,pool_x_img),0)
        if 'rna' in data_type:
            x_rna = self.rna_gnn_2(x_rna,edge_index_rna) 
            x_rna = self.rna_relu_2(x_rna)   
            batch = torch.zeros(len(x_rna),dtype=torch.long).to(device)
            pool_x_rna,att_rna_2,att_rna_each = self.mpool_rna(x_rna,batch)
            att_2.append(att_rna_2)
            att_each_2.append(att_rna_each)
            pool_x = torch.cat((pool_x,pool_x_rna),0)
        if 'cli' in data_type:
            x_cli = self.cli_gnn_2(x_cli,edge_index_cli) 
            x_cli = self.cli_relu_2(x_cli)   
            batch = torch.zeros(len(x_cli),dtype=torch.long).to(device)
            pool_x_cli,att_cli_2,att_cli_each = self.mpool_cli(x_cli,batch)
            att_2.append(att_cli_2)
            att_each_2.append(att_cli_each)
            pool_x = torch.cat((pool_x,pool_x_cli),0)

        fea_dict['mae_labels'] = pool_x


        if len(train_use_type)>1:
            if use_type == train_use_type:
                mae_x = self.mae(pool_x,mask).squeeze(0)
                fea_dict['mae_out'] = mae_x
            else:
                k=0
                tmp_x = torch.zeros((len(train_use_type),pool_x.size(1))).to(device)
                mask = np.ones(len(train_use_type),dtype=bool)
                for i,type_ in enumerate(train_use_type):
                    if type_ in data_type:
                        tmp_x[i] = pool_x[k]
                        k+=1
                        mask[i] = False
                mask = np.expand_dims(mask,0)
                mask = np.expand_dims(mask,0)
                if k==0:
                    mask = np.array([[[False]*len(train_use_type)]])
                mae_x = self.mae(tmp_x,mask).squeeze(0)
                fea_dict['mae_out'] = mae_x   


            save_fea['after_mae'] = mae_x.cpu().detach().numpy() 
            if mix:
                mae_x = self.mix(mae_x)
                save_fea['after_mix'] = mae_x.cpu().detach().numpy() 

            k=0
            if 'img' in train_use_type and 'img' in use_type:
                x_img = x_img + mae_x[train_use_type.index('img')] 
                k+=1
            if 'rna' in train_use_type and 'rna' in use_type:
                x_rna = x_rna + mae_x[train_use_type.index('rna')]  
                k+=1
            if 'cli' in train_use_type and 'cli' in use_type:
                x_cli = x_cli + mae_x[train_use_type.index('cli')]  
                k+=1
            
 
        att_3 = []
        pool_x = torch.empty((0)).to(device)

        
        if 'img' in data_type:
            batch = torch.zeros(len(x_img),dtype=torch.long).to(device)
            pool_x_img,att_img_3,_ = self.mpool_img_2(x_img,batch)
            att_3.append(att_img_3)

            pool_x = torch.cat((pool_x,pool_x_img),0)
        if 'rna' in data_type:
            batch = torch.zeros(len(x_rna),dtype=torch.long).to(device)
            pool_x_rna,att_rna_3,_ = self.mpool_rna_2(x_rna,batch)
            att_3.append(att_rna_3)

            pool_x = torch.cat((pool_x,pool_x_rna),0)
        if 'cli' in data_type:
            batch = torch.zeros(len(x_cli),dtype=torch.long).to(device)
            pool_x_cli,att_cli_3,_ = self.mpool_cli_2(x_cli,batch)
            att_3.append(att_cli_3)

            pool_x = torch.cat((pool_x,pool_x_cli),0) 
            

        
        x = pool_x
        x = F.normalize(x, dim=1)
        fea = x
        
        k=0
        if 'img' in data_type:
            fea_dict['img'] = fea[k]
            k+=1
        if 'rna' in data_type:
            fea_dict['rna'] = fea[k]       
            k+=1
        if 'cli' in data_type:
            fea_dict['cli'] = fea[k]
            k+=1

        
        k=0
        multi_x = torch.empty((0)).to(device)

        if 'img' in data_type:
            x_img = self.lin1_img(x[k])
            x_img = self.relu(x_img)
            x_img = self.norm_img(x_img)
            x_img = self.dropout(x_img)    

            x_img = self.lin2_img(x_img).unsqueeze(0)
            # x_img = self.attention(x_img, batch) # 1
            multi_x = torch.cat((multi_x,x_img),0)
            k+=1
        if 'rna' in data_type:
            x_rna = self.lin1_rna(x[k])
            x_rna = self.relu(x_rna)
            x_rna = self.norm_rna(x_rna)
            x_rna = self.dropout(x_rna) 

            x_rna = self.lin2_rna(x_rna).unsqueeze(0)
            # x_rna = self.attention(x_rna, batch) # 2
            multi_x = torch.cat((multi_x,x_rna),0)
            k+=1
        if 'cli' in data_type:
            x_cli = self.lin1_cli(x[k])
            x_cli = self.relu(x_cli)
            x_cli = self.norm_cli(x_cli)
            x_cli = self.dropout(x_cli)

            x_cli = self.lin2_rna(x_cli).unsqueeze(0)
            # x_cli = self.attention(x_cli, batch)#3
            multi_x = torch.cat((multi_x,x_cli),0)

            k+=1


        one_x = torch.mean(multi_x, dim=0)

        return (one_x,multi_x),save_fea,(att_2,att_3,att_each_2),fea_dict   # 返回multi_x平均值、保存特征，注意力2，3，特征字典












