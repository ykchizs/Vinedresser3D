from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .serialized_attn import SerializeMode, sparse_serialized_scaled_dot_product_self_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from ...attention import RotaryPositionEmbedder

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)            
        return (x * self.gamma * self.scale).to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[SparseTensor, torch.Tensor], num_fused: int) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _rope(self, qkv: SparseTensor) -> SparseTensor:
        q, k, v = qkv.feats.unbind(dim=1)   # [T, H, C]
        q, k = self.rope(q, k, qkv.coords[:, 1:])
        qkv = qkv.replace(torch.stack([q, k, v], dim=1)) 
        return qkv
    
    def forward(self, x: Union[SparseTensor, torch.Tensor], context: Optional[Union[SparseTensor, torch.Tensor]] = None, inject_values: dict = None) -> Union[SparseTensor, torch.Tensor]:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope:
                qkv = self._rope(qkv)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "serialized":
                h = sparse_serialized_scaled_dot_product_self_attention(
                    qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
                )
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                raise ValueError("stage2: qk_rms_norm is True")
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))
            
            k, v = kv.unbind(dim=2)

            # injection
            # if inject_values is not None and inject_values['inject'] == "extract" and inject_values['block_id']>28:
            #     value_name = str(inject_values['t']) + '_' + str(inject_values['second_order']) + '_' + str(inject_values['block_id']) +  '_' + 'Value'
            #     inject_values['values'][value_name] = v.cpu()
            #     print(f"Extract {value_name}")
            # if inject_values is not None and inject_values['inject'] == "inject" and inject_values['block_id']>28:
            #     value_name = str(inject_values['t']) + '_' + str(inject_values['second_order']) + '_' + str(inject_values['block_id']) +  '_' + 'Value'
            #     if value_name in inject_values['values'].keys():
            #         v = inject_values['values'][value_name].cuda()
            #         print(f"Inject {value_name}")
            #     else: 
            #         raise ValueError(f"Value {value_name} not found in inject_values['values']")

            # get the gt attention map
            # print("block_id: ", inject_values['block_id'])
            # atten_map = torch.zeros(q.feats.shape[0], k.shape[1]).to(q.feats.device)
            # for i in range(k.shape[1]):
            #     v1 = torch.zeros_like(v).to(q.feats.device)
            #     v1[0,i,0,0] = 1
            #     q1, k1 = q, k
            #     result = sparse_scaled_dot_product_attention(q1, k1, v1)
            #     atten_map[:,i] = result.feats[:,0,0]
            # inject_values[f'block{inject_values["block_id"]}'] = atten_map

            h = sparse_scaled_dot_product_attention(q, kv)
            # torch.save(q.feats, f"outputs/features_PCA/owl_feats/owl_block{inject_values['block_id']}_feats.pt")
            # torch.save(q.coords, f"outputs/features_PCA/owl_feats/owl_block{inject_values['block_id']}_coords.pt")
            # h = sparse_scaled_dot_product_attention(q, k, v)

            # if (inject_values['t_id'] in [0,4,8,12,16,20,24]) and inject_values['second_order'] == False and inject_values['block_id'] in [0,6,12,18,23] and inject_values['cond_type'] == "cond":
            # if inject_values['second_order'] == False and inject_values['cond_type'] == "cond":
            # atten_name = str(inject_values['t_id']) + '_' + str(inject_values['second_order']) + '_' + str(inject_values['block_id']) +  '_' + 'atten'
            # dot_product = torch.einsum('nwh,bmwh->bnm', q.feats, k) / ((k.shape[-2]*k.shape[-1])**0.5)
            # values = dot_product[0,:,0]
            # create_attention_visualization(values, q.coords[:,1:], "outputs/attention_visualization/animalcar_cat", f"block{inject_values['block_id']}")
            
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h
