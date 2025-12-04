import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

# yichuan 1031

"""
<cls>
<pad>
<eos>
<unk>
L
A
G
V
S
E
R
T
I
D
P
K
Q
N
F
Y
M
H
W
C
X
B
U
Z
O
.
-
<null_1>
<mask>
"""


class MLPMixPrompt(nn.Module):
    """
    用 MLP 生成每个 prompt 位置的 over-vocab logits，再 softmax 混合词表嵌入。
    可选：最后接一个低秩投影到 vocab，显著降参。
    """
    def __init__(
        self,
        embed_weight: torch.Tensor,     # ESM embed_tokens.weight: [V, H]
        prompt_len: int,
        num_tasks: int,
        task_dim: int = 128,            # 任务向量维度
        hidden: int = 256,              # MLP 隐藏维度
        temperature: float = 1.0,
        mask_token_ids: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.num_tasks = int(num_tasks)
        self.temperature = max(1e-6, float(temperature))
        self.embed_weight_ref = embed_weight  # 引用 ESM 词表
        self.vocab_size, self.hidden_size = embed_weight.shape

        if mask_token_ids is None:
            mask_token_ids = []
        if not isinstance(mask_token_ids, list):
            raise TypeError("mask_token_ids must be a list of ints or None.")

        # 检查是否越界 越界就报错
        for i in mask_token_ids:
            if not (0 <= i < self.vocab_size):
                raise ValueError(f"mask_token_id {i} out of range [0, {self.vocab_size})")

        self.mask_token_ids = mask_token_ids  # 就存原样的 list

        # 每个 task 一个可学习向量（也可以改成来自外部输入）
        self.task_embed = nn.Embedding(num_tasks, task_dim)

        layers = [nn.Linear(task_dim, hidden)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden))
        layers += [nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        out_dim = self.prompt_len * self.vocab_size
        layers.append(nn.Linear(hidden, out_dim))
        self.mlp = nn.Sequential(*layers)

        # 初始化
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _task_idxs(self, task_ids: torch.Tensor) -> torch.Tensor:
        device = self.embed_weight_ref.device
        task_ids = task_ids.to(device).long()

        if task_ids.ndim == 1:
            return task_ids

        if task_ids.ndim == 2:
            return task_ids.max(dim=1).values

        raise ValueError(f"Unexpected task_ids shape: {tuple(task_ids.shape)}")

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        输入:
            task_ids: [B] / [B, L]（行内非零为唯一 task id，其余位置为 0）
        输出:
            prompts: [B, P, H]
        """
        # 统一拿到 [B]
        idx = self._task_idxs(task_ids)                      # [B]

        t = self.task_embed(idx)                             # [B, task_dim]
        out = self.mlp(t)                                    # [B, P*V]

        logits_b = out.view(-1, self.prompt_len, self.vocab_size)  # [B,P,V]

        if self.mask_token_ids:
            logits_b = logits_b.clone()  # 避免 in place 影响 autograd
            for tok_id in self.mask_token_ids:
                logits_b[..., tok_id] = float("-inf")

        weights = torch.softmax(logits_b / self.temperature, dim=-1)   # [B,P,V]
        prompts = weights.matmul(self.embed_weight_ref)                # [B,P,H]
        return prompts, weights
