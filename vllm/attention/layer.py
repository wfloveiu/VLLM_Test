"""Attention layer."""
from typing import List, Optional

import ray._private.worker
import torch
import torch.nn as nn

from vllm.attention.backends.abstract import (AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.selector import get_attn_backend

from vllm.output_qkv import write_tensor

import os
import ray
class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """
    # 不想修改接口
    layer = 0 #哪一个layer层
    output_gate = False # 因为在推理前会有warmup，不想统计warmup的数据
    output_dir = "QKV_Output/llama2-chat-hf/csv"
    generate_tokens = 0
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backend = get_attn_backend(torch.get_default_dtype())
        impl_cls = self.backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window)
   
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata[AttentionMetadataPerStage],
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        
        if query.shape[0] == 6:
            Attention.output_gate = True
        if Attention.output_gate:
            if ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE:
                write_tensor(query, Attention.output_dir,Attention.layer,"output_ray_q")
                write_tensor(key, Attention.output_dir,Attention.layer,"output_ray_k")
                write_tensor(value,Attention.output_dir,Attention.layer,"output_ray_v")
            else:
                write_tensor(query, Attention.output_dir,Attention.layer,"output_driver_q")
                write_tensor(key, Attention.output_dir,Attention.layer,"output_driver_k")
                write_tensor(value,Attention.output_dir,Attention.layer,"output_driver_v")
            Attention.layer = Attention.layer + 1
            if Attention.layer == 32:
                Attention.layer = 0
                Attention.generate_tokens+=1
                print("generate ", Attention.generate_tokens,"tokens\n")
                print("-----------------------------------------------------------------")
        
        
        return self.impl.forward(query, key, value, kv_cache, attn_metadata,
                                 kv_scale)

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        return s
