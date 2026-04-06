# GuppyLM 环境搭建与聊天指南

## 项目简介

GuppyLM 是一个约 8.7M 参数的极简 LLM，扮演一条叫 Guppy 的小鱼，用短句聊水、食物、鱼缸生活等话题。

- 参数量：8.7M
- 架构：6 层 Vanilla Transformer
- 词表：4096（BPE）
- 最大序列长度：128 tokens
- 模型文件大小：约 34MB

## 环境搭建步骤

### 第 1 步：进入项目目录

```bash
cd /path/to/guppylm
```

### 第 2 步：创建 Python 虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> 所有依赖会安装在项目目录下的 `.venv/` 文件夹中，不污染系统 Python。

### 第 3 步：安装依赖

```bash
pip install torch tokenizers tqdm numpy datasets huggingface_hub "httpx[socks]"
```

| 包 | 作用 |
|---|---|
| `torch` | PyTorch，模型训练和推理 |
| `tokenizers` | BPE 分词器，文本与 token 互转 |
| `tqdm` | 进度条显示 |
| `numpy` | 数值计算 |
| `datasets` | 加载 HuggingFace 数据集 |
| `huggingface_hub` | 从 HuggingFace 下载模型 |
| `httpx[socks]` | SOCKS 代理支持（网络需要时） |

### 第 4 步：下载预训练模型

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='arman-bd/guppylm-9M', local_dir='checkpoints')
"
```

模型会下载到 `checkpoints/` 目录，包含以下文件：

```
checkpoints/
├── pytorch_model.bin    # 34MB，模型权重
├── tokenizer.json       # 160KB，BPE 分词器
├── config.json          # 模型架构配置
├── model.py             # 模型定义代码
├── inference.py         # 推理代码
├── config.py            # 配置类
└── ...
```

### 第 5 步：准备文件路径

代码默认从以下路径读取文件，需要复制一份：

```bash
mkdir -p data
cp checkpoints/tokenizer.json data/tokenizer.json
```

> `pytorch_model.bin` 可以直接用，不需要重命名。`inference.py` 同时支持纯 state_dict 和带 metadata 的 checkpoint 格式。

## 与模型聊天

### 方式一：交互式命令行（推荐）

```bash
source .venv/bin/activate
python -m guppylm chat
```

启动后效果：

```
Guppy Chat (type 'quit' to exit)

You> hi guppy
Guppy> hi there. i just found a nice spot near the rock.

You> are you hungry
Guppy> yes. always yes. i will swim to the top right now.

You> quit
```

输入 `quit` / `exit` / `q` 退出。

### 方式二：Python 脚本

```python
from guppylm.inference import GuppyInference

# 加载模型
engine = GuppyInference(
    'checkpoints/pytorch_model.bin',  # 模型权重路径
    'data/tokenizer.json',            # 分词器路径
    device='cpu'                      # 也可改为 'cuda' 如果有 GPU
)

# 单轮对话
r = engine.chat_completion([{'role': 'user', 'content': 'hi guppy'}])
print(r['choices'][0]['message']['content'])

# 多轮对话（带历史上下文）
msgs = []
msgs.append({'role': 'user', 'content': 'hi guppy'})
r = engine.chat_completion(msgs)
reply = r['choices'][0]['message']['content']
print(f'Guppy> {reply}')
msgs.append({'role': 'assistant', 'content': reply})

msgs.append({'role': 'user', 'content': 'are you hungry'})
r = engine.chat_completion(msgs)
print(f'Guppy> {r["choices"][0]["message"]["content"]}')
```

### 方式三：命令行参数

```bash
source .venv/bin/activate
python -m guppylm.inference --checkpoint checkpoints/pytorch_model.bin --tokenizer data/tokenizer.json --device cpu
```

## 完整一键脚本

```bash
cd /path/to/guppylm
python3 -m venv .venv
source .venv/bin/activate
pip install torch tokenizers tqdm numpy datasets huggingface_hub "httpx[socks]"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='arman-bd/guppylm-9M', local_dir='checkpoints')"
mkdir -p data && cp checkpoints/tokenizer.json data/tokenizer.json
python -m guppylm chat
```

## 常见问题

### 下载慢 / 网络问题

如果下载失败，可能是网络代理问题。确保安装了 `httpx[socks]`，或者设置代理：

```bash
export HTTP_PROXY=socks5://127.0.0.1:7890
export HTTPS_PROXY=socks5://127.0.0.1:7890
```

### 模型在 CPU 上慢吗

8.7M 参数非常小，CPU 上单次回复约 1-2 秒，完全可用。

### 想自己训练模型

运行以下命令（需要 GPU，推荐 Colab T4）：

```bash
# 生成 60K 训练数据 + 训练分词器
python -m guppylm prepare

# 训练模型（10000 步，约 5 分钟）
python -m guppylm train

# 用训练好的模型聊天
python -m guppylm chat
```

## 文件结构速览

```
guppylm/
├── guppylm/              # 核心代码包
│   ├── config.py         # 模型 + 训练超参数
│   ├── model.py          # Transformer 架构（129 行）
│   ├── dataset.py        # 数据加载与批处理
│   ├── train.py          # 训练循环
│   ├── inference.py      # 聊天推理
│   ├── generate_data.py  # 60 主题模板数据生成
│   └── prepare_data.py   # 数据准备 + tokenizer 训练
├── checkpoints/          # 预训练模型（下载后）
├── data/                 # 训练数据和分词器
├── docs/                 # 文档
├── train_guppylm.ipynb   # Colab 训练笔记本
├── use_guppylm.ipynb     # Colab 聊天笔记本
└── requirements.txt      # 依赖列表
```
