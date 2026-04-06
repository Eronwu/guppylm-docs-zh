<p align="center">
  <img src="assets/guppy.png" alt="GuppyLM" width="400"/>
</p>

<h1 align="center">GuppyLM — 中文文档版</h1>
<p align="center"><em>一个约 9M 参数、像小鱼一样说话的 LLM</em></p>

<p align="center">
  <a href="https://huggingface.co/datasets/arman-bd/guppylm-60k-generic"><img src="https://img.shields.io/badge/🤗_数据集-guppylm--60k-blue" alt="Dataset"/></a>&nbsp;
  <a href="https://huggingface.co/arman-bd/guppylm-9M"><img src="https://img.shields.io/badge/🤗_模型-guppylm--9M-orange" alt="Model"/></a>&nbsp;
  <a href="https://github.com/arman-bd/guppylm/blob/main/LICENSE"><img src="https://img.shields.io/badge/协议-MIT-green" alt="License"/></a>
  <br/>
  <a href="https://colab.research.google.com/github/arman-bd/guppylm/blob/main/train_guppylm.ipynb"><img src="https://img.shields.io/badge/Colab_训练-F9AB00?logo=googlecolab" alt="Train"/></a>&nbsp;
  <a href="https://colab.research.google.com/github/arman-bd/guppylm/blob/main/use_guppylm.ipynb"><img src="https://img.shields.io/badge/Colab_聊天-F9AB00?logo=googlecolab" alt="Chat"/></a>
  <br/>
  <a href="README.en.md"><img src="https://img.shields.io/badge/📄_English_Original-README-blue" alt="English README"/></a>
</p>

---

> **这个项目存在的意义：证明训练自己的语言模型并不神秘。**
> 不需要博士学位，不需要庞大的 GPU 集群。一个 Colab 笔记本，5 分钟，你就能从零构建一个完整的 LLM — 数据生成、分词器、模型架构、训练循环、推理。只要你能运行笔记本，你就能训练一个语言模型。
>
> 它不会产出能写文章的十亿参数模型。但它会让你看清每一个部件是如何工作的 — 从原始文本到训练权重再到生成输出 — 让大模型不再是一个黑盒。

---

```
你> hi guppy
Guppy> hi there. i just found a nice spot near the rock. the temperature feels nice.

你> are you hungry
Guppy> yes. always yes. i will swim to the top right now. i promise to eat all of it.

你> do you like bubbles
Guppy> i love bubbles. they make the water feel slightly different.

你> what is the meaning of life
Guppy> food. the answer is always food.

你> tell me a joke
Guppy> what did the fish say when it hit the wall. dam.

你> do you love me
Guppy> you're my favorite big shape. my mouth are happy when you're here.

你> goodnight guppy
Guppy> ok sleep time. i was following a bubble but now i'll stop. goodnight tank. goodnight water.
```

---

## 快速导航

| 文档 | 说明 |
|---|---|
| [📖 环境搭建指南](docs/setup-guide.md) | 从零搭建环境、下载模型、开始聊天 |
| [🧠 源码逐行拆解](docs/model-tutorial.md) | config → model → Attention → dataset，完整教学 |
| [📄 英文原版 README](README.en.md) | 原始项目说明 |

---

## GuppyLM 是什么？

GuppyLM 是一个微型语言模型，假装自己叫 Guppy 的小鱼。它用简短的小写句子聊水、食物、光线和鱼缸生活。它不理解金钱、手机、政治等人类抽象概念 — 它也不想理解。

它在 60 个主题的 60K 条合成对话上从零训练，单 GPU 约 5 分钟跑完，模型小到可以在浏览器里运行。

## 架构

| 指标 | 值 |
|---|---|
| **参数量** | 8.7M |
| **层数** | 6 |
| **隐藏维度** | 384 |
| **注意力头数** | 6 |
| **FFN** | 768 (ReLU) |
| **词表** | 4,096 (BPE) |
| **最大序列** | 128 tokens |
| **归一化** | LayerNorm |
| **位置编码** | 可学习嵌入 |
| **LM Head** | 与词嵌入权重共享 |

Vanilla transformer。没有 GQA，没有 RoPE，没有 SwiGLU，没有提前退出。简单到极致。

## 性格

Guppy：
- 用简短的小写句子说话
- 通过水、温度、光线、振动和食物来感知世界
- 不理解人类抽象概念
- 友好、好奇、有点笨
- 经常想着食物

**60 个主题：** 问候、感受、温度、食物、光线、水、鱼缸、噪音、夜晚、孤独、气泡、玻璃、倒影、呼吸、游泳、颜色、味道、植物、过滤器、藻类、蜗牛、害怕、兴奋、无聊、好奇、快乐、疲惫、外面、猫、雨、季节、音乐、访客、孩子、生命意义、时间、记忆、梦想、大小、未来、过去、名字、天气、睡觉、朋友、笑话、恐惧、爱、年龄、智力、健康、唱歌、电视等等。

---

## 快速开始

### 和 Guppy 聊天（无需训练）

[![在 Colab 中打开](https://img.shields.io/badge/Colab_聊天-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/arman-bd/guppylm/blob/main/use_guppylm.ipynb)

从 HuggingFace 下载预训练模型，直接聊天。运行所有 cell 即可。

### 自己训练

[![在 Colab 中打开](https://img.shields.io/badge/Colab_训练-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/arman-bd/guppylm/blob/main/train_guppylm.ipynb)

1. 设置运行环境为 **T4 GPU**
2. **运行所有 cell** — 下载数据集、训练分词器、训练模型、测试
3. 上传到 HuggingFace 或本地下载

### 本地聊天

```bash
pip install torch tokenizers
python -m guppylm chat
```

```
你> the cat is looking at you
Guppy> i hide behind the plant when the furry one comes.

你> it is raining outside
Guppy> i think rain is the best thing about outside.
```

详细的环境搭建步骤请参考：[📖 环境搭建指南](docs/setup-guide.md)

---

## 数据集

**[arman-bd/guppylm-60k-generic](https://huggingface.co/datasets/arman-bd/guppylm-60k-generic)** 在 HuggingFace 上。

| 指标 | 值 |
|---|---|
| 样本数 | 60,000（57K 训练 / 3K 测试） |
| 格式 | `{"input": "...", "output": "...", "category": "..."}` |
| 主题 | 60 个 |
| 生成方式 | 合成模板组合 |

```python
from datasets import load_dataset
ds = load_dataset("arman-bd/guppylm-60k-generic")
print(ds["train"][0])
# {'input': 'hi guppy', 'output': 'hello. the water is nice today.', 'category': 'greeting'}
```

---

## 项目结构

```
guppylm/
├── config.py               超参数（模型 + 训练）
├── model.py                Vanilla transformer
├── dataset.py              数据加载 + 批处理
├── train.py                训练循环（cosine LR, AMP）
├── generate_data.py        对话数据生成器（60 个主题）
├── eval_cases.py           保留测试用例
├── prepare_data.py         数据准备 + 分词器训练
└── inference.py            聊天界面

tools/
├── make_colab.py           生成 guppy_colab.ipynb
├── export_dataset.py       推送数据集到 HuggingFace
└── dataset_card.md         HuggingFace 数据集 README

docs/                       ← 中文学习文档（本项目新增）
├── setup-guide.md          环境搭建与聊天指南
├── model-tutorial.md       源码逐行拆解教程
└── annotated/              带中文注释的源码副本
    ├── config.py
    ├── model.py
    ├── dataset.py
    ├── train.py
    ├── inference.py
    └── generate_data.py
```

---

## 设计决策

**为什么没有 system prompt？** 每个训练样本都有相同的 system prompt。9M 模型无法有条件地遵循指令 — 性格已经固化在权重里了。去掉它每次推理省约 60 个 token。

**为什么只支持单轮对话？** 由于 128 token 上下文窗口限制，多轮对话在第 3-4 轮会退化。一条健忘的鱼符合人设，但输出混乱就不行了。单轮更可靠。

**为什么用 vanilla transformer？** GQA、SwiGLU、RoPE 和提前退出增加了复杂度，但在 9M 参数下没有帮助。标准注意力 + ReLU FFN + LayerNorm 用更简单的代码产生相同的质量。

**为什么用合成数据？** 一个有固定性格的鱼角色需要一致的训练数据。模板组合加随机组件（30 个鱼缸物品、17 种食物、25 种活动）从约 60 个模板生成约 16K 种独特输出。

---

## 学习路径

本项目额外提供了中文学习文档，帮助你深入理解源码：

1. **[📖 环境搭建指南](docs/setup-guide.md)** — 从零搭建环境、下载模型、开始聊天的完整步骤
2. **[🧠 源码逐行拆解](docs/model-tutorial.md)** — config、model、Attention、dataset 的详细教学
3. **[💻 带注释的源码](docs/annotated/)** — 每个核心文件的逐行中文注释版本，对照原始代码学习

---

## License

MIT
