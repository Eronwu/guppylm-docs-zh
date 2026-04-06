# GuppyLM 源码逐行拆解教程

## 目录

1. [项目总览](#1-项目总览)
2. [config.py — 所有数字的含义](#2-configpy--所有数字的含义)
3. [model.py — 模型骨架](#3-modelpy--模型骨架)
4. [Attention 深度拆解](#4-attention-深度拆解)
5. [dataset.py — 数据怎么变成训练对](#5-datasetpy--数据怎么变成训练对)

---

## 1. 项目总览

### 1.1 GuppyLM 是什么

GuppyLM 是一个约 **8.7M 参数**的极简语言模型，扮演一条叫 Guppy 的小鱼，用短句聊水、食物、鱼缸生活等话题。

| 指标 | 值 |
|---|---|
| 参数量 | 8.7M |
| 层数 | 6 |
| 隐藏维度 | 384 |
| 注意力头数 | 6 |
| FFN 隐藏维度 | 768 |
| 词表大小 | 4,096 (BPE) |
| 最大序列长度 | 128 tokens |
| 归一化 | LayerNorm (Pre-Norm) |
| 位置编码 | 可学习嵌入 |
| LM Head | 与词嵌入权重共享 |

### 1.2 架构特点

> Vanilla transformer. No GQA, no SwiGLU, no parallel residual, no RoPE. As simple as it gets.

- **没有** 分组查询注意力（GQA）
- **没有** SwiGLU 激活
- **没有** 并行残差
- **没有** 旋转位置编码（RoPE）
- 就是最原始的 Transformer，适合学习

### 1.3 项目文件结构

```
guppylm/
├── guppylm/              # 核心代码包
│   ├── config.py         # 模型 + 训练超参数（36 行）
│   ├── model.py          # Transformer 架构（129 行）
│   ├── dataset.py        # 数据加载与批处理（51 行）
│   ├── train.py          # 训练循环（161 行）
│   ├── inference.py      # 聊天推理（124 行）
│   ├── generate_data.py  # 60 主题模板数据生成
│   ├── prepare_data.py   # 数据准备 + tokenizer 训练
│   └── eval_cases.py     # 测试用例
├── checkpoints/          # 预训练模型
├── data/                 # 训练数据和分词器
├── train_guppylm.ipynb   # Colab 训练笔记本
└── use_guppylm.ipynb     # Colab 聊天笔记本
```

### 1.4 整体数据流

```
用户输入 "hi guppy"
  ↓
Tokenizer 编码 → [15, 234, 891]          # 文本 → token IDs
  ↓
Embedding → 384 维向量                    # token IDs → 稠密向量
  ↓
6 × Transformer Block                     # 核心计算
  ↓
LM Head → [4096, 4096, 4096, ...]        # 每个位置对每个 token 打分
  ↓
Softmax + 采样 → 下一个 token             # 生成
  ↓
循环直到遇到结束标记或达到最大长度
  ↓
解码 → "hi there. i'm under the log."     # token IDs → 文本
```

---

## 2. config.py — 所有数字的含义

完整代码（36 行）：

```python
from dataclasses import dataclass

@dataclass
class GuppyConfig:
    vocab_size: int = 4096
    max_seq_len: int = 128
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_hidden: int = 768
    dropout: float = 0.1

    # Special tokens
    pad_id: int = 0
    bos_id: int = 1           # 消息开始标记
    eos_id: int = 2           # 消息结束标记

@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 10000
    eval_interval: int = 200
    save_interval: int = 500
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "checkpoints"
```

### 2.1 模型参数详解

| 参数 | 值 | 含义 | 类比 |
|---|---|---|---|
| `vocab_size` | 4096 | 词表大小：模型认识 4096 个 token | 字典里有多少个字/词 |
| `max_seq_len` | 128 | 最大序列长度 | 一次能读多长的句子 |
| `d_model` | 384 | 隐藏层维度 | 每个词的"理解深度" |
| `n_layers` | 6 | Transformer 层数 | 思考的"层数" |
| `n_heads` | 6 | 注意力头数 | 同时关注几个"方面" |
| `ffn_hidden` | 768 | 前馈网络隐藏维度 | 每层的"脑容量"（= d_model × 2） |
| `dropout` | 0.1 | 随机丢弃率 | 防止过拟合的"遗忘"机制 |

### 2.2 特殊 Token

| ID | Token | 作用 |
|---|---|---|
| 0 | `<pad>` | 填充：不同长度句子对齐时用 |
| 1 | 开始标记 | 消息开始 |
| 2 | 结束标记 | 消息结束 |

### 2.3 训练参数详解

| 参数 | 值 | 含义 |
|---|---|---|
| `batch_size` | 32 | 每次训练 32 条数据 |
| `learning_rate` | 3e-4 | 初始学习率 |
| `min_lr` | 3e-5 | 最小学习率（cosine 衰减终点） |
| `weight_decay` | 0.1 | L2 正则化强度 |
| `warmup_steps` | 200 | 前 200 步线性预热 |
| `max_steps` | 10000 | 总共训练 10000 步 |
| `eval_interval` | 200 | 每 200 步评估一次 |
| `save_interval` | 500 | 每 500 步保存一次检查点 |
| `grad_clip` | 1.0 | 梯度裁剪阈值，防止梯度爆炸 |

### 2.4 参数量计算

```
词嵌入:     vocab_size × d_model     = 4096 × 384  = 1,572,864
位置嵌入:   max_seq_len × d_model    = 128 × 384   = 49,152

每层 Attention:
  QKV:      3 × d_model²             = 3 × 384²   = 442,368
  Output:   d_model²                 = 384²       = 147,456
  小计:                                589,824

每层 FFN:
  Up:       d_model × ffn_hidden     = 384 × 768  = 294,912
  Down:     ffn_hidden × d_model     = 768 × 384  = 294,912
  小计:                                589,824

6 层总计:   6 × (589,824 + 589,824)             = 7,078,848

LayerNorm:  每层 2 个 × 6 层 × d_model × 2      = 9,216
LM Head:    与词嵌入权重共享，不额外计算

总计:       1,572,864 + 49,152 + 7,078,848 + 9,216 ≈ 8.7M
```

---

## 3. model.py — 模型骨架

完整代码（129 行），分 5 个类。

### 3.1 整体结构

```
GuppyLM
├── tok_emb          # Token Embedding (4096 → 384)
├── pos_emb          # Position Embedding (128 → 384)
├── drop             # Dropout
├── blocks           # 6 × Transformer Block
│   ├── Block 1
│   │   ├── norm1    # LayerNorm
│   │   ├── attn     # Multi-Head Attention
│   │   ├── norm2    # LayerNorm
│   │   └── ffn      # Feed-Forward Network
│   ├── Block 2
│   ├── ...
│   └── Block 6
├── norm             # Final LayerNorm
└── lm_head          # Linear (384 → 4096, 权重共享)
```

### 3.2 Attention 类（第 15-34 行）

```python
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads                    # 6
        self.head_dim = config.d_model // config.n_heads # 384 // 6 = 64

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)  # 384 → 1152
        self.out = nn.Linear(config.d_model, config.d_model)      # 384 → 384
        self.dropout = nn.Dropout(config.dropout)
```

**关键设计：QKV 合并**

```python
# GuppyLM 的做法：一个矩阵同时算 Q, K, V
self.qkv = nn.Linear(384, 3 * 384)

# 等价但更慢的写法：
self.q = nn.Linear(384, 384)
self.k = nn.Linear(384, 384)
self.v = nn.Linear(384, 384)
```

一个矩阵做一次大乘法 vs 三个矩阵各做一次小乘法，数学等价，但 GPU 对大矩阵乘法优化更好。

**forward 方法：**

```python
def forward(self, x, mask=None):
    B, T, C = x.shape  # (batch, seq_len, dim) = (32, 128, 384)

    # 第 1 步：算 Q, K, V
    qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 (B, 6, T, 64)

    # 第 2 步：注意力分数
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

    # 第 3 步：因果掩码
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))

    # 第 4 步：softmax + dropout
    attn = self.dropout(F.softmax(attn, dim=-1))

    # 第 5 步：加权 V + 输出投影
    return self.out((attn @ v).transpose(1, 2).contiguous().view(B, T, C))
```

### 3.3 FFN 类（第 37-45 行）

```python
class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_hidden)    # 384 → 768
        self.down = nn.Linear(config.ffn_hidden, config.d_model)  # 768 → 384
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.relu(self.up(x))))
```

**作用：** Attention 负责"看别人"，FFN 负责"自己思考"。

数据流：`384维 → 768维 → ReLU → 384维`

为什么先升再降？给模型更大的空间做非线性变换，再压缩回去。这是 Transformer 的标准设计，ffn_hidden 通常是 d_model 的 2 倍（GPT-3 是 4 倍）。

### 3.4 Block 类（第 48-59 行）

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)   # Pre-Norm + 残差
        x = x + self.ffn(self.norm2(x))          # Pre-Norm + 残差
        return x
```

**两个关键概念：**

1. **Pre-Norm**：先 LayerNorm 再进子层。比 Post-Norm 训练更稳定，梯度更流畅。
2. **残差连接**（`x + ...`）：让梯度能直接流到前面，防止深层网络梯度消失。

一个 Block 的完整流程：
```
输入 x
  → LayerNorm → Attention → + x (残差)
  → LayerNorm → FFN → + x (残差)
  → 输出
```

### 3.5 GuppyLM 类（第 62-104 行）

**初始化：**

```python
class GuppyLM(nn.Module):
    def __init__(self, config: GuppyConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)  # 4096 → 384
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model) # 128 → 384
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 权重共享！

        self.apply(self._init_weights)
```

**权重共享（Weight Tying）：**

`lm_head` 和 `tok_emb` 用同一个权重矩阵。好处：
- 省 1.5M 参数
- 输入和输出用同一套"词义理解"
- 训练更稳定，尤其在小模型上效果明显

**权重初始化：**

```python
def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
```

标准的小方差正态初始化（std=0.02），来自 GPT-2 论文。

**forward 方法：**

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # 词 + 位置
    mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)

    for block in self.blocks:
        x = block(x, mask)

    logits = self.lm_head(self.norm(x))

    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
            ignore_index=0,  # 忽略 pad token
        )

    return logits, loss
```

**因果掩码（Causal Mask）：**

```
位置:  0  1  2  3
0     [1  0  0  0]   ← 位置 0 只能看到自己
1     [1  1  0  0]   ← 位置 1 能看到 0 和 1
2     [1  1  1  0]   ← 位置 2 能看到 0,1,2
3     [1  1  1  1]   ← 位置 3 能看到全部
```

保证生成时"只能看到前面的词，不能偷看后面的"。

### 3.6 generate 方法（第 106-121 行）

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens=64, temperature=0.7, top_k=50, **kwargs):
    self.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.config.max_seq_len:]   # 截断到最大长度
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature        # 取最后一个 token

        # Top-K 采样
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        if next_id.item() == self.config.eos_id:       # 遇到结束标记停止
            break
    return idx, []
```

**生成过程（自回归）：**

```
输入: "hi guppy" → [15, 234, 891]
  → 预测下一个 → 采样得到 "hi" → [15, 234, 891, 42]
  → 预测下一个 → 采样得到 "there" → [15, 234, 891, 42, 108]
  → 预测下一个 → 采样得到 "." → [15, 234, 891, 42, 108, 7]
  → ... 遇到结束标记停止
输出: "hi there."
```

**温度的作用：**
- `temperature < 1`（如 0.7）：分布更尖锐，输出更确定/保守
- `temperature = 1`：原始概率分布
- `temperature > 1`：分布更平坦，输出更随机/有创意

**Top-K 采样：** 只保留概率最大的 K 个 token，其余设为 -inf。防止模型采样到极低概率的奇怪 token。

---

## 4. Attention 深度拆解

### 4.1 什么是 Q、K、V？

用图书馆查书类比：

| 概念 | 类比 | 作用 |
|---|---|---|
| **Query（查询）** | 你想找什么书 | 当前 token 的"需求" |
| **Key（键）** | 每本书的标签 | 其他 token 的"特征" |
| **Value（值）** | 书的内容 | 其他 token 的"实际信息" |

每个 token 都有 Q、K、V 三个向量：
1. 用我的 **Q** 去和所有 token 的 **K** 做匹配（点积）
2. 匹配度越高，说明那个 token 和我越相关
3. 按相关程度加权组合所有 token 的 **V**

### 4.2 维度变化全过程

假设输入 `x` 形状为 `(32, 10, 384)`：

```python
# 第 1 步：一次算出 Q, K, V
qkv = self.qkv(x)                    # (32, 10, 1152)  ← 384 → 3×384

# 第 2 步：拆成 3 份，再分头
qkv = qkv.reshape(32, 10, 3, 6, 64)  # (B, T, 3, 6头, 每头64维)
qkv = qkv.permute(2, 0, 3, 1, 4)     # (3, B, 6头, T, 64)
q, k, v = qkv[0], qkv[1], qkv[2]     # 每个都是 (32, 6, 10, 64)
```

### 4.3 注意力分数计算

以序列 `"hi guppy ."` 为例（3 个 token）：

```
Q @ K^T 结果 (注意力分数矩阵):
        hi    guppy    .
hi    [ 5.2    3.1    0.8 ]   ← "hi" 最关注自己(5.2)，其次 "guppy"(3.1)
guppy [ 2.8    6.4    1.2 ]   ← "guppy" 最关注自己(6.4)，其次 "hi"(2.8)
.     [ 1.5    2.2    4.1 ]   ← "." 最关注自己(4.1)，但也关注 "guppy"(2.2)
```

除以 `sqrt(64) = 8` 缩放后，再过 softmax：

```
Softmax 后 (注意力权重):
        hi    guppy    .
hi    [ 0.95   0.00   0.00 ]   ← "hi" 几乎只看自己
guppy [ 0.12   0.88   0.00 ]   ← "guppy" 主要看自己，也看 "hi"
.     [ 0.15   0.25   0.60 ]   ← "." 看自己最多，也看前面两个
```

### 4.4 加权 V

```
output_hi   = 0.95 × v_hi   + 0.00 × v_guppy + 0.00 × v_.
output_guppy = 0.12 × v_hi  + 0.88 × v_guppy + 0.00 × v_.
output_.     = 0.15 × v_hi  + 0.25 × v_guppy + 0.60 × v_.
```

### 4.5 为什么叫"多头"？

6 个头意味着模型同时从 6 个不同角度理解句子：

```
句子: "the fish in the water is happy"

头 1 关注语法:  fish → is (主谓关系)
头 2 关注修饰:  fish → in the water (位置关系)
头 3 关注情感:  is → happy (状态关系)
头 4 关注指代:  the → fish (限定关系)
头 5 关注全局:  所有 token 平均关注
头 6 关注局部:  相邻 token 关注
```

每个头学到的东西不同，拼起来就是更丰富的理解。

### 4.6 Attention 完整流程图

```
输入 x (32, 10, 384)
  │
  ▼
┌─────────────────────────────┐
│  Linear(384 → 1152)         │  ← 一个矩阵算 Q, K, V
│  reshape → permute → split  │
└─────────────────────────────┘
  │
  ├─ q (32, 6, 10, 64)
  ├─ k (32, 6, 10, 64)
  └─ v (32, 6, 10, 64)
  │
  ▼
┌─────────────────────────────┐
│  q @ k^T / sqrt(64)         │  ← 注意力分数 (32, 6, 10, 10)
│  masked_fill(-inf)          │  ← 遮住未来
│  softmax                    │  ← 转概率
│  @ v                        │  ← 加权求和
└─────────────────────────────┘
  │
  ▼
  (32, 6, 10, 64)
  │
  ▼
┌─────────────────────────────┐
│  transpose → view           │  ← 拼回 (32, 10, 384)
│  Linear(384 → 384)          │  ← 输出投影
└─────────────────────────────┘
  │
  ▼
输出 (32, 10, 384)
```

---

## 5. dataset.py — 数据怎么变成训练对

完整代码（51 行）：

```python
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


class GuppyDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, max_len: int = 512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.samples = []

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                ids = self.tokenizer.encode(data["text"]).ids
                if len(ids) > max_len:
                    ids = ids[:max_len]
                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = ids[:-1]   # 去掉最后一个
        y = ids[1:]    # 去掉第一个
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
    return padded_x, padded_y


def get_dataloader(path, tokenizer_path, max_len=512, batch_size=32, shuffle=True):
    dataset = GuppyDataset(path, tokenizer_path, max_len)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
```

### 5.1 原始数据格式

每条数据是 JSONL 格式（一行一个 JSON）：

```json
{"text": "用户消息\nhi guppy\n助手回复\nhello. the water is nice today.\n", "category": "greeting"}
```

### 5.2 数据加载流程

```
train.jsonl (57K 行)
  │
  ▼ 逐行读取
{"text": "...", "category": "..."}
  │
  ▼ Tokenizer 编码
[1, 15, 234, 891, 2, 1, 42, 108, 7, 2]  ← token IDs
  │
  ▼ 截断（如果超过 max_len）
[1, 15, 234, 891, 2, 1, 42, 108, 7, 2]  ← 最多 128 个
  │
  ▼ 存入 samples
samples = [[...], [...], ...]  ← 57K 条
```

### 5.3 训练对生成（核心）

LLM 训练的本质是**预测下一个 token**：

```python
def __getitem__(self, idx):
    ids = self.samples[idx]        # [1, 15, 234, 891, 2, 1, 42, 108, 7, 2]
    x = ids[:-1]                   # [1, 15, 234, 891, 2, 1, 42, 108, 7]   ← 输入
    y = ids[1:]                    # [15, 234, 891, 2, 1, 42, 108, 7, 2]   ← 目标
    return x, y
```

**原理：**

```
输入 x:  [1,  15,  234,  891,  2,  1,  42,  108,  7]
目标 y:  [15, 234,  891,  2,   1,  42, 108,  7,   2]
         ↑    ↑     ↑     ↑    ↑   ↑    ↑     ↑    ↑
模型要学会：给定位置 i 的所有前面 token，预测位置 i+1 的 token
```

这就是自回归语言模型的核心思想。

### 5.4 批处理（collate_fn）

不同句子长度不同，需要 padding 对齐：

```python
def collate_fn(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)  # 找 batch 内最长序列

    # 创建全 pad 的张量
    padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)

    # 填入实际数据
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y

    return padded_x, padded_y
```

**例子：**

```
batch 内有 3 条数据：
  x1 = [1, 15, 234]           (长度 3)
  x2 = [1, 42, 108, 7, 2]     (长度 5)
  x3 = [1, 891]               (长度 2)

padding 后（max_len=5）：
  padded_x = [
    [1, 15, 234, 0, 0],       ← 用 0 (pad_id) 填充
    [1, 42, 108, 7, 2],
    [1, 891, 0, 0, 0],
  ]
```

损失函数里 `ignore_index=0` 会忽略这些 pad 位置，不参与训练。

### 5.5 DataLoader

```python
def get_dataloader(path, tokenizer_path, max_len=512, batch_size=32, shuffle=True):
    dataset = GuppyDataset(path, tokenizer_path, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,      # 每次 32 条
        shuffle=True,               # 打乱顺序
        collate_fn=collate_fn,      # 自定义批处理
        num_workers=0,              # 主进程加载（简单但慢）
        pin_memory=True,            # 锁页内存，加速 GPU 传输
    )
```

### 5.6 完整数据流

```
train.jsonl (57K 行)
  │
  ▼ GuppyDataset
samples = [[token_ids...], [token_ids...], ...]  (57K 条)
  │
  ▼ __getitem__
x = ids[:-1], y = ids[1:]  (训练对)
  │
  ▼ DataLoader (batch_size=32)
collate_fn → padding 对齐
  │
  ▼
x: (32, max_len_in_batch)   ← 输入
y: (32, max_len_in_batch)   ← 目标
  │
  ▼ 送入模型
loss = cross_entropy(model(x), y, ignore_index=0)
```
