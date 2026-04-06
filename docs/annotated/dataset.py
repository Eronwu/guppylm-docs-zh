"""GuppyLM 数据集加载模块。

负责将 JSONL 格式的文本数据转换为模型可训练的 (input, target) 批次。

核心概念：LLM 训练的本质是"预测下一个 token"。
对于序列 [a, b, c, d, e]：
  - 输入 x = [a, b, c, d]
  - 目标 y = [b, c, d, e]
  - 模型学会：给定当前位置及之前的所有 token，预测下一个 token
"""

import json

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


class GuppyDataset(Dataset):
    """GuppyLM 数据集。

    读取 JSONL 文件，每行格式：
        {"text": "用户消息\nhi guppy\n助手回复\nhello.\n", "category": "greeting"}

    加载后自动完成：
    1. Tokenizer 编码：文本 → token IDs
    2. 截断：超过 max_len 的部分丢弃
    3. 过滤：太短（<2 token）的样本丢弃
    """

    def __init__(self, path: str, tokenizer_path: str, max_len: int = 512):
        """初始化数据集。

        Args:
            path: JSONL 文件路径（train.jsonl 或 eval.jsonl）
            tokenizer_path: 分词器文件路径（tokenizer.json）
            max_len: 最大序列长度，超过则截断
        """
        # 加载预训练好的 BPE 分词器
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.samples = []  # 存储所有样本的 token ID 列表

        # 逐行读取 JSONL 文件
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                # 将文本编码为 token IDs，例如 "hi guppy" → [15, 234, 891]
                ids = self.tokenizer.encode(data["text"]).ids

                # 截断：如果序列太长，只保留前面部分
                if len(ids) > max_len:
                    ids = ids[:max_len]

                # 过滤：至少需要 2 个 token 才能形成 (x, y) 训练对
                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self):
        """返回数据集样本数量。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取单个训练对。

        LLM 训练的核心：预测下一个 token。

        假设 ids = [1, 15, 234, 891, 2, 1, 42, 108, 7, 2]

        输入 x = ids[:-1] = [1, 15, 234, 891, 2, 1, 42, 108, 7]   # 去掉最后一个
        目标 y = ids[1:]  = [15, 234, 891, 2, 1, 42, 108, 7, 2]   # 去掉第一个

        模型学会：
          位置 0 看到 [1]       → 预测 15
          位置 1 看到 [1, 15]   → 预测 234
          位置 2 看到 [1,15,234] → 预测 891
          ...

        Args:
            idx: 样本索引
        Returns:
            (input_ids, target_ids) 两个张量
        """
        ids = self.samples[idx]
        x = ids[:-1]   # 输入：去掉最后一个 token
        y = ids[1:]    # 目标：去掉第一个 token（相当于 x 整体左移一位）
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    """自定义批处理函数：将不同长度的序列 padding 对齐。

    不同句子长度不同，一个 batch 内需对齐到相同长度才能组成张量。
    策略：以 batch 内最长序列为准，短的用 pad_id (0) 填充。

    示例：
        batch 内有 3 条：
          x1 = [1, 15, 234]           (长度 3)
          x2 = [1, 42, 108, 7, 2]     (长度 5)
          x3 = [1, 891]               (长度 2)

        padding 后（max_len=5）：
          padded_x = [
            [1, 15, 234, 0, 0],       ← 用 0 填充
            [1, 42, 108, 7, 2],
            [1, 891, 0, 0, 0],
          ]

    Args:
        batch: list of (x, y) 元组
        pad_id: 填充 token 的 ID，默认 0
    Returns:
        (padded_x, padded_y) 两个对齐后的张量
    """
    xs, ys = zip(*batch)
    # 找到 batch 内最长序列的长度
    max_len = max(len(x) for x in xs)

    # 创建全 pad 的张量
    padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)

    # 将实际数据填入
    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y

    return padded_x, padded_y


def get_dataloader(path, tokenizer_path, max_len=512, batch_size=32, shuffle=True):
    """创建 DataLoader。

    完整数据流：
        JSONL 文件 → GuppyDataset → __getitem__ 生成 (x, y)
        → DataLoader 按 batch_size 分组 → collate_fn padding 对齐
        → 输出 (batch_x, batch_y) 送入模型

    Args:
        path: JSONL 文件路径
        tokenizer_path: 分词器文件路径
        max_len: 最大序列长度
        batch_size: 批次大小
        shuffle: 是否打乱顺序（训练时 True，评估时 False）
    Returns:
        PyTorch DataLoader
    """
    dataset = GuppyDataset(path, tokenizer_path, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,      # 每次取 32 条数据
        shuffle=shuffle,            # 训练时打乱，保证每个 batch 数据分布均匀
        collate_fn=collate_fn,      # 自定义批处理：padding 对齐
        num_workers=0,              # 使用主进程加载（简单，适合小项目）
        pin_memory=True,            # 锁页内存：加速 CPU → GPU 数据传输
    )
