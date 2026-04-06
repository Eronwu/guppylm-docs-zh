# 带中文注释的源码

这个目录包含 GuppyLM 所有核心代码的中文注释版本。

每行关键代码都有详细的中文注释，适合边读边学。

## 文件列表

| 文件 | 行数 | 内容 |
|---|---|---|
| [config.py](config.py) | 36 行 | 模型和训练的所有超参数 |
| [model.py](model.py) | 129 行 | Transformer 架构（Attention、FFN、Block、GuppyLM） |
| [dataset.py](dataset.py) | 51 行 | 数据加载、训练对生成、批处理 |
| [train.py](train.py) | 161 行 | 训练循环、学习率调度、AMP、评估保存 |
| [inference.py](inference.py) | 124 行 | 模型加载、prompt 格式化、聊天生成 |
| [generate_data.py](generate_data.py) | 精简版 | 60 主题模板数据生成原理说明（原始文件 1700 行，主要是模板数据） |

## 阅读顺序建议

```
1. config.py       ← 先看，了解所有数字的含义
2. model.py        ← 核心，理解 transformer 怎么工作
3. dataset.py      ← 理解数据怎么变成训练对
4. train.py        ← 理解训练流程
5. inference.py    ← 理解推理怎么工作
6. generate_data.py ← 了解训练数据怎么来的
```

## 与原始代码对照

这些文件是原始代码的**副本 + 中文注释**，原始代码在 `guppylm/` 目录下保持干净不动。

建议对照阅读：左边看原始代码，右边看注释版。
