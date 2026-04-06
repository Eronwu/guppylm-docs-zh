"""GuppyLM 模型定义 — 一条小鱼的大脑。

这是一个最简 vanilla transformer 实现：
- 多头自注意力（Multi-Head Self-Attention）
- ReLU 激活的前馈网络（FFN）
- Pre-Norm LayerNorm + 残差连接
- 可学习的位置编码（Learned Positional Embeddings）

没有 GQA、没有 SwiGLU、没有并行残差、没有 RoPE。
简单到极致，适合学习。

完整架构：8.7M 参数，6 层，384 维，6 头，4096 词表。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GuppyConfig


class Attention(nn.Module):
    """多头自注意力层。

    核心思想：每个 token 通过 Query 去和其他 token 的 Key 做匹配，
    按匹配程度（注意力分数）加权组合所有 token 的 Value。

    优化：Q、K、V 三个线性层合并为一个，一次矩阵乘法算出，GPU 效率更高。

    输入形状: (batch, seq_len, d_model) = (32, 128, 384)
    输出形状: (batch, seq_len, d_model) = (32, 128, 384)
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads                          # 注意力头数 = 6
        self.head_dim = config.d_model // config.n_heads       # 每个头维度 = 384/6 = 64

        # 一个线性层同时计算 Q, K, V（384 → 3×384 = 1152）
        # 比分开写三个 Linear(384→384) 更高效
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)   # 输出投影层
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        """前向传播。

        Args:
            x: 输入张量 (B, T, C) = (batch, seq_len, d_model)
            mask: 因果掩码，遮住未来 token（下三角矩阵）

        Returns:
            注意力输出 (B, T, C)
        """
        B, T, C = x.shape

        # ── 第 1 步：一次算出 Q, K, V ─────────────────────────
        # qkv 形状: (B, T, 3*C) = (32, 128, 1152)
        # reshape 后: (B, T, 3, n_heads, head_dim) = (32, 128, 3, 6, 64)
        # permute 后: (3, B, n_heads, T, head_dim) — 把 3 提到最前面方便拆分
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 (B, n_heads, T, head_dim) = (32, 6, 128, 64)

        # ── 第 2 步：计算注意力分数 ─────────────────────────────
        # q @ k^T: (32, 6, 128, 64) @ (32, 6, 64, 128) → (32, 6, 128, 128)
        # 每个头独立计算：每个 token 和其他所有 token 的点积
        # 除以 sqrt(head_dim) 防止数值过大导致 softmax 梯度消失
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (32, 6, 128, 128)

        # ── 第 3 步：因果掩码（遮住未来）─────────────────────────
        # mask 是下三角矩阵，对角线以上为 0
        # masked_fill 把 0 的位置填 -inf，softmax 后变成 0
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # ── 第 4 步：Softmax 转概率 + Dropout ───────────────────
        # 最后一维做 softmax：每行加起来 = 1，变成注意力权重
        attn = self.dropout(F.softmax(attn, dim=-1))

        # ── 第 5 步：加权 V + 输出投影 ──────────────────────────
        # attn @ v: (32, 6, 128, 128) @ (32, 6, 128, 64) → (32, 6, 128, 64)
        # transpose(1,2): (32, 128, 6, 64)
        # contiguous().view: 合并头和维度 → (32, 128, 384)
        # self.out: 最后过一层线性层混合信息
        return self.out((attn @ v).transpose(1, 2).contiguous().view(B, T, C))


class FFN(nn.Module):
    """前馈网络（Feed-Forward Network）。

    结构：Linear → ReLU → Linear → Dropout
    维度：d_model → ffn_hidden → d_model = 384 → 768 → 384

    Attention 负责"看别人"（token 间交互），
    FFN 负责"自己思考"（每个 token 独立变换）。
    """

    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_hidden)     # 升维：384 → 768
        self.down = nn.Linear(config.ffn_hidden, config.d_model)   # 降维：768 → 384
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入 (B, T, d_model)
        Returns:
            输出 (B, T, d_model)
        """
        # up(x): 先升维到 768，给非线性变换更大的空间
        # F.relu: 激活函数，引入非线性
        # down(): 降回 384
        # dropout: 随机丢弃防止过拟合
        return self.dropout(self.down(F.relu(self.up(x))))


class Block(nn.Module):
    """一个 Transformer 层（Block）。

    结构：Pre-Norm + 残差连接
    流程：x → LayerNorm → Attention → +x → LayerNorm → FFN → +x

    Pre-Norm（先归一化再进子层）比 Post-Norm 训练更稳定，
    残差连接（x + ...）让梯度能直接流到前面，防止梯度消失。
    """

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)    # Attention 前的归一化
        self.attn = Attention(config)                # 多头自注意力
        self.norm2 = nn.LayerNorm(config.d_model)    # FFN 前的归一化
        self.ffn = FFN(config)                       # 前馈网络

    def forward(self, x, mask=None):
        """前向传播。

        Args:
            x: 输入 (B, T, d_model)
            mask: 因果掩码
        Returns:
            输出 (B, T, d_model)
        """
        # 残差分支 1：归一化 → 注意力 → 加回原始输入
        x = x + self.attn(self.norm1(x), mask)
        # 残差分支 2：归一化 → FFN → 加回原始输入
        x = x + self.ffn(self.norm2(x))
        return x


class GuppyLM(nn.Module):
    """GuppyLM 完整模型。

    架构：
        Token Embedding + Position Embedding → Dropout
        → 6 × Transformer Block
        → LayerNorm → LM Head

    权重共享：LM Head 和 Token Embedding 用同一个矩阵，
    省 1.5M 参数，训练更稳定。
    """

    def __init__(self, config: GuppyConfig):
        super().__init__()
        self.config = config

        # ── 嵌入层 ────────────────────────────────────────────
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)    # 词嵌入：token ID → 384 维
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)   # 位置嵌入：位置 0~127 → 384 维
        self.drop = nn.Dropout(config.dropout)                            # Dropout

        # ── Transformer 层 ────────────────────────────────────
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])  # 6 个 Block

        # ── 输出层 ────────────────────────────────────────────
        self.norm = nn.LayerNorm(config.d_model)                                         # 最终归一化
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)         # 语言模型头：384 → 4096
        self.lm_head.weight = self.tok_emb.weight  # 权重共享：LM Head 和词嵌入用同一个矩阵

        # 初始化所有权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化。

        使用小方差正态分布（std=0.02），来自 GPT-2 论文。
        小方差让初始输出接近 0，训练更稳定。
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """前向传播。

        Args:
            idx: 输入 token IDs，形状 (B, T)
            targets: 目标 token IDs，形状 (B, T)。训练时传入用于计算 loss。

        Returns:
            logits: 每个位置对每个 token 的打分 (B, T, vocab_size)
            loss: 交叉熵损失（仅当 targets 不为 None 时）
        """
        B, T = idx.shape

        # ── 第 1 步：嵌入 ─────────────────────────────────────
        # pos: [0, 1, 2, ..., T-1] 位置索引
        pos = torch.arange(T, device=idx.device)
        # tok_emb(idx): 把 token ID 映射为 384 维向量 (B, T, 384)
        # pos_emb(pos): 把位置映射为 384 维向量 (T, 384)，广播到 (B, T, 384)
        # 两者相加得到带位置信息的词向量
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # ── 第 2 步：因果掩码 ─────────────────────────────────
        # torch.tril 生成下三角矩阵（对角线及以下为 1，以上为 0）
        # 形状: (T, T) → unsqueeze 两次 → (1, 1, T, T)，广播到 (B, n_heads, T, T)
        # 作用：让每个 token 只能看到自己和前面的 token
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

        # ── 第 3 步：过所有 Transformer 层 ─────────────────────
        for block in self.blocks:
            x = block(x, mask)

        # ── 第 4 步：最终归一化 + 输出 ──────────────────────────
        logits = self.lm_head(self.norm(x))  # (B, T, vocab_size)

        # ── 第 5 步：计算损失（训练时）─────────────────────────
        loss = None
        if targets is not None:
            # logits: (B, T, vocab_size) → (B*T, vocab_size)
            # targets: (B, T) → (B*T,)
            # ignore_index=0: 忽略 pad token，不参与损失计算
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=0,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.7, top_k=50, **kwargs):
        """自回归文本生成。

        原理：每次预测下一个 token，拼到输入后面，循环直到遇到结束标记。

        Args:
            idx: 输入 token IDs (B, T)
            max_new_tokens: 最多生成多少个 token
            temperature: 温度参数。<1 更确定，>1 更随机
            top_k: 只保留概率最大的 K 个 token

        Returns:
            生成的完整 token IDs (B, T+new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 只取最后 max_seq_len 个 token（防止超过模型最大长度）
            idx_cond = idx[:, -self.config.max_seq_len:]

            # 前向传播，取最后一个位置的 logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-K 采样：只保留概率最大的 K 个，其余设为 -inf
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # 转概率分布 + 随机采样
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 拼到输入后面
            idx = torch.cat([idx, next_id], dim=1)

            # 遇到结束标记就停止
            if next_id.item() == self.config.eos_id:
                break
        return idx, []

    def param_count(self):
        """统计参数量。

        Returns:
            (总参数量, 0) — 第二个值占位，预留给非训练参数统计
        """
        total = sum(p.numel() for p in self.parameters())
        return total, 0

    def param_summary(self):
        """返回参数量摘要字符串。

        例如: "GuppyLM: 8,700,000 params (8.7M)"
        """
        total, _ = self.param_count()
        return f"GuppyLM: {total:,} params ({total/1e6:.1f}M)"
