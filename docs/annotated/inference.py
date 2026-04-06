"""GuppyLM 推理模块 — 聊天接口。

负责加载训练好的模型和分词器，将用户输入转换为 prompt，
调用模型生成回复，再解码为文本返回。

支持两种使用方式：
1. 作为 Python 模块导入：from guppylm.inference import GuppyInference
2. 命令行直接运行：python -m guppylm.inference
"""

import json
import time
import uuid

import torch
from tokenizers import Tokenizer

from .config import GuppyConfig
from .model import GuppyLM


class GuppyInference:
    """GuppyLM 推理引擎。

    加载模型 + 分词器，提供 chat_completion 接口。
    兼容多种 checkpoint 格式：
    - 训练保存的格式：{"model_state_dict": ..., "config": ...}
    - HuggingFace 格式：纯 state_dict + 外部 config.json
    """

    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        """初始化推理引擎。

        Args:
            checkpoint_path: 模型权重文件路径（.pt 或 .bin）
            tokenizer_path: 分词器文件路径（tokenizer.json）
            device: 计算设备（"cpu" / "cuda" / "mps"）
        """
        self.device = torch.device(device)
        # 加载 BPE 分词器
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        import os
        # 加载模型权重
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # ── 提取 state_dict ──────────────────────────────────
        # 兼容两种格式：
        # 1. 训练保存的：{"model_state_dict": {...}, "config": {...}}
        # 2. HuggingFace 的：纯 state_dict（直接是参数字典）
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # ── 加载模型配置 ─────────────────────────────────────
        # 优先级：外部 config.json > checkpoint 内嵌 config > 默认值
        config_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        config_path = os.path.join(config_dir, "config.json")

        if os.path.exists(config_path):
            # 从外部 config.json 读取（HuggingFace 格式）
            with open(config_path) as f:
                cfg = json.load(f)
            # 兼容 HuggingFace 标准键名和 GuppyLM 自有键名
            self.config = GuppyConfig(
                vocab_size=cfg.get("vocab_size", 4096),
                max_seq_len=cfg.get("max_position_embeddings", cfg.get("max_seq_len", 128)),
                d_model=cfg.get("hidden_size", cfg.get("d_model", 384)),
                n_layers=cfg.get("num_hidden_layers", cfg.get("n_layers", 6)),
                n_heads=cfg.get("num_attention_heads", cfg.get("n_heads", 6)),
                ffn_hidden=cfg.get("intermediate_size", cfg.get("ffn_hidden", 768)),
                dropout=cfg.get("hidden_dropout_prob", cfg.get("dropout", 0.1)),
                pad_id=cfg.get("pad_token_id", cfg.get("pad_id", 0)),
                bos_id=cfg.get("bos_token_id", cfg.get("bos_id", 1)),
                eos_id=cfg.get("eos_token_id", cfg.get("eos_id", 2)),
            )
        elif isinstance(ckpt, dict) and "config" in ckpt:
            # 从 checkpoint 内嵌的 config 读取（训练保存的格式）
            valid_fields = {f.name for f in GuppyConfig.__dataclass_fields__.values()}
            self.config = GuppyConfig(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
        else:
            # 都没有则用默认值
            print("Warning: No config found, using defaults")
            self.config = GuppyConfig()

        # ── 创建模型并加载权重 ────────────────────────────────
        self.model = GuppyLM(self.config).to(self.device)
        # 过滤掉不匹配的 key（防止加载时有额外的 key 报错）
        filtered = {k: v for k, v in state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(filtered)
        self.model.eval()  # 切换到评估模式（关闭 dropout）

        total, _ = self.model.param_count()
        print(f"GuppyLM loaded: {total/1e6:.1f}M params")

    def chat_completion(self, messages, temperature=0.7, max_tokens=64,
                        top_k=50, **kwargs):
        """聊天补全接口。

        将消息列表格式化为 prompt，模型生成回复后返回。
        返回格式兼容 OpenAI Chat Completion API。

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "hi"}, ...]
            temperature: 温度参数，控制生成随机性
            max_tokens: 最多生成多少个 token
            top_k: Top-K 采样，只保留概率最大的 K 个 token

        Returns:
            {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
        """
        # ── 第 1 步：格式化 prompt ───────────────────────────
        prompt = self._format_prompt(messages)

        # ── 第 2 步：编码为 token IDs ────────────────────────
        input_ids = self.tokenizer.encode(prompt).ids
        prompt_tokens = len(input_ids)  # 记录 prompt 长度，后面用来截取新生成的部分
        input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # ── 第 3 步：模型生成 ────────────────────────────────
        output_t, _ = self.model.generate(input_t, max_tokens, temperature, top_k)

        # ── 第 4 步：解码为文本 ──────────────────────────────
        # 只取新生成的部分（跳过 prompt 对应的 token）
        output_text = self.tokenizer.decode(output_t[0].tolist()[prompt_tokens:])

        # ── 第 5 步：清理输出 ────────────────────────────────
        # 截断第一个 </think>：防止模型泄漏到下一轮
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[0]
        # 也截断 <think>：防止模型重复输出角色标记
        if "<think>" in output_text:
            output_text = output_text.split("<think>")[0]
        resp_text = output_text.strip()

        # 返回 OpenAI 兼容格式
        return {
            "choices": [{
                "message": {"role": "assistant", "content": resp_text},
            }],
        }

    def _format_prompt(self, messages):
        """将消息列表格式化为 ChatML 格式的 prompt。

        格式：
            <think>user
            hi guppy
            </think>
            <think>assistant
            <模型从这里开始生成>

        注意：system 消息被忽略，因为 GuppyLM 的性格已经固化在权重里，
        不需要 system prompt 来引导。

        Args:
            messages: [{"role": "user", "content": "..."}, ...]
        Returns:
            格式化后的 prompt 字符串
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            # 忽略 system 消息
            if role == "system":
                continue
            parts.append(f"<think>{role}\n{content}</think>")
        # 最后加上 assistant 开头，引导模型生成回复
        parts.append("<think>assistant\n")
        return "\n".join(parts)


def main():
    """命令行聊天入口。

    使用方式：
        python -m guppylm.inference
        python -m guppylm.inference --checkpoint checkpoints/best_model.pt --device cpu
    """
    import argparse
    p = argparse.ArgumentParser(description="Chat with Guppy")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    engine = GuppyInference(args.checkpoint, args.tokenizer, args.device)
    print("\nGuppy Chat (type 'quit' to exit)")
    msgs = []
    while True:
        inp = input("\nYou> ").strip()
        if inp.lower() in ("quit", "exit", "q"):
            break
        msgs.append({"role": "user", "content": inp})
        result = engine.chat_completion(msgs)
        msg = result["choices"][0]["message"]
        if msg.get("content"):
            print(f"Guppy> {msg['content']}")
        msgs.append(msg)


if __name__ == "__main__":
    main()
