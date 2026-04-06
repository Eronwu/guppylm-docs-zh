"""GuppyLM 配置模块

定义模型架构和训练过程的所有超参数。
整个项目只有这一个地方需要改数字。
"""

from dataclasses import dataclass


@dataclass
class GuppyConfig:
    """模型架构配置。

    这是一个 8.7M 参数的极简 transformer：
    - 6 层，每层 6 个注意力头
    - 隐藏维度 384，FFN 隐藏层 768
    - BPE 分词器，词表大小 4096
    - 最大序列长度 128 tokens
    """
    # ── 模型架构 ──────────────────────────────────────────────
    vocab_size: int = 4096     # 词表大小：模型认识 4096 个 token（子词级别）
    max_seq_len: int = 128     # 最大序列长度：一次最多处理 128 个 token
    d_model: int = 384         # 隐藏层维度：每个 token 用 384 维向量表示
    n_layers: int = 6          # Transformer 层数：6 个 Block 串联
    n_heads: int = 6           # 多头注意力的头数：6 个独立视角
    ffn_hidden: int = 768      # 前馈网络隐藏维度：d_model 的 2 倍
    dropout: float = 0.1       # Dropout 概率：训练时随机丢弃 10% 的神经元

    # ── 特殊 Token ──────────────────────────────────────────
    pad_id: int = 0            # 填充标记：不同长度句子 padding 对齐时用
    bos_id: int = 1            # 消息开始标记（<think>）
    eos_id: int = 2            # 消息结束标记（</think>）


@dataclass
class TrainConfig:
    """训练过程配置。

    训练策略：
    - AdamW 优化器，初始学习率 3e-4
    - 前 200 步线性 warmup，之后 cosine 衰减到 3e-5
    - 梯度裁剪 1.0 防止梯度爆炸
    - 总共 10000 步，每 200 步评估，每 500 步保存
    """
    # ── 优化器 ──────────────────────────────────────────────
    batch_size: int = 32           # 批次大小：每次训练 32 条数据
    learning_rate: float = 3e-4    # 初始学习率（warmup 结束后达到）
    min_lr: float = 3e-5           # 最小学习率（cosine 衰减的终点）
    weight_decay: float = 0.1      # L2 正则化权重衰减系数

    # ── 学习率调度 ──────────────────────────────────────────
    warmup_steps: int = 200        # 预热步数：前 200 步从 0 线性增长到 lr
    max_steps: int = 10000         # 总训练步数

    # ── 评估与保存 ──────────────────────────────────────────
    eval_interval: int = 200       # 评估间隔：每 200 步在验证集上评估
    save_interval: int = 500       # 保存间隔：每 500 步保存一次检查点
    grad_clip: float = 1.0         # 梯度裁剪阈值：防止梯度爆炸

    # ── 运行时 ──────────────────────────────────────────────
    device: str = "auto"           # 设备选择：auto 自动检测 cuda/mps/cpu
    seed: int = 42                 # 随机种子：保证实验可复现
    data_dir: str = "data"         # 数据目录：存放 train.jsonl, eval.jsonl, tokenizer.json
    output_dir: str = "checkpoints"  # 输出目录：存放模型检查点和配置
