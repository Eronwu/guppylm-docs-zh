"""GuppyLM 训练循环。

完整的训练流程：
1. 初始化模型、优化器、数据加载器
2. 学习率调度：warmup + cosine 衰减
3. 训练循环：前向 → 反向 → 梯度裁剪 → 更新
4. 定期评估 + 保存最佳模型
5. 支持 AMP（自动混合精度）加速

训练策略：
- AdamW 优化器（权重衰减 + 动量）
- 前 200 步线性 warmup（学习率从 0 增长到峰值）
- 之后 cosine 衰减到最小学习率
- 梯度裁剪 1.0 防止梯度爆炸
- 每 200 步评估，每 500 步保存
"""

import json
import math
import os
import time

import torch

from .config import GuppyConfig, TrainConfig
from .dataset import get_dataloader
from .model import GuppyLM


def get_device(config):
    """自动选择可用的计算设备。

    优先级：CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

    Args:
        config: 训练配置，含 device 字段
    Returns:
        torch.device 对象
    """
    if config.device == "auto":
        # 1. 优先检查 NVIDIA GPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        # 2. 其次检查 Apple Silicon (M1/M2/M3)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        # 3. 最后回退到 CPU
        return torch.device("cpu")
    # 用户手动指定了设备
    return torch.device(config.device)


def get_lr(step, config):
    """计算当前步的学习率。

    学习率调度策略 = 线性 warmup + Cosine 衰减：

    阶段 1（warmup）：step < warmup_steps
        lr = learning_rate × (step / warmup_steps)
        从 0 线性增长到 learning_rate

    阶段 2（cosine 衰减）：step >= warmup_steps
        lr = min_lr + (learning_rate - min_lr) × 0.5 × (1 + cos(π × progress))
        从 learning_rate 平滑衰减到 min_lr

    为什么需要 warmup？
    - 训练初期梯度不稳定，大学习率容易发散
    - 小学习率起步让模型先"热身"，找到好的方向

    为什么用 cosine 衰减？
    - 比线性衰减更平滑
    - 后期小学习率让模型精细调整权重

    Args:
        step: 当前训练步数
        config: 训练配置
    Returns:
        当前步的学习率
    """
    # ── 阶段 1：线性 warmup ──────────────────────────────────
    if step < config.warmup_steps:
        # step=0 时 lr=0，step=warmup_steps 时 lr=learning_rate
        return config.learning_rate * step / config.warmup_steps

    # ── 阶段 2：Cosine 衰减 ──────────────────────────────────
    # progress: 0 → 1，表示 warmup 结束到训练结束的比例
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    # cosine 系数：1 → 0（从 cos(0)=1 到 cos(π)=-1，映射到 0.5*(1+cos) 从 1 到 0）
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    # 从 learning_rate 衰减到 min_lr
    return config.min_lr + (config.learning_rate - config.min_lr) * coeff


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    """在验证集上评估模型。

    计算平均 loss 作为评估指标。loss 越低说明模型越好。

    Args:
        model: 模型
        loader: 验证集 DataLoader
        device: 计算设备
        max_batches: 最多评估多少个 batch（避免评估太慢）
    Returns:
        平均 loss
    """
    model.eval()  # 切换到评估模式（关闭 dropout）
    total_loss, n = 0, 0
    for x, y in loader:
        if n >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()  # 切回训练模式
    return total_loss / max(1, n)


def train():
    """主训练函数。

    完整流程：
    1. 初始化：模型、优化器、数据加载器
    2. 训练循环：
       a. 计算当前学习率
       b. 前向传播 → 计算 loss
       c. 反向传播 → 计算梯度
       d. 梯度裁剪 → 防止梯度爆炸
       e. 优化器更新 → 更新权重
    3. 定期评估（每 eval_interval 步）
    4. 保存最佳模型（eval loss 最低时）
    5. 定期保存检查点（每 save_interval 步）
    """
    # ── 第 1 步：初始化配置 ──────────────────────────────────
    mc = GuppyConfig()     # 模型配置
    tc = TrainConfig()     # 训练配置
    device = get_device(tc)
    torch.manual_seed(tc.seed)  # 设置随机种子，保证可复现

    print(f"Device: {device}")

    # ── 第 2 步：创建模型 ────────────────────────────────────
    tokenizer_path = os.path.join(tc.data_dir, "tokenizer.json")
    model = GuppyLM(mc).to(device)
    print(model.param_summary())

    # ── 第 3 步：创建数据加载器 ──────────────────────────────
    # 训练集：shuffle=True，打乱数据
    train_loader = get_dataloader(
        os.path.join(tc.data_dir, "train.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, shuffle=True,
    )
    # 验证集：shuffle=False，保持顺序
    eval_loader = get_dataloader(
        os.path.join(tc.data_dir, "eval.jsonl"), tokenizer_path,
        mc.max_seq_len, tc.batch_size, shuffle=False,
    )
    print(f"Train: {len(train_loader.dataset):,}, Eval: {len(eval_loader.dataset):,}")

    # ── 第 4 步：创建优化器 ──────────────────────────────────
    # AdamW = Adam + 权重衰减（解耦的 L2 正则化）
    # betas=(0.9, 0.95): 一阶动量衰减 0.9，二阶动量衰减 0.95
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.learning_rate,
        weight_decay=tc.weight_decay, betas=(0.9, 0.95),
    )

    # ── 第 5 步：AMP（自动混合精度）设置 ─────────────────────
    # 只在 CUDA 上启用 AMP，用 float16 加速计算、减少显存
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── 第 6 步：创建输出目录 + 保存配置 ──────────────────────
    os.makedirs(tc.output_dir, exist_ok=True)
    with open(os.path.join(tc.output_dir, "config.json"), "w") as f:
        json.dump({"model": vars(mc), "train": vars(tc)}, f, indent=2)

    # ── 第 7 步：训练循环 ────────────────────────────────────
    model.train()
    step, best_eval = 0, float("inf")  # best_eval 初始为无穷大
    losses = []                         # 记录训练 loss
    t0 = time.time()                    # 记录开始时间

    # 打印表头
    print(f"\nTraining for {tc.max_steps} steps...")
    print(f"{'Step':>6} | {'LR':>10} | {'Train':>10} | {'Eval':>10} | {'Time':>8}")
    print("-" * 56)

    # 外层 while：确保遍历完所有数据后还能继续（如果一轮不够 max_steps）
    while step < tc.max_steps:
        # 内层 for：遍历训练集
        for x, y in train_loader:
            if step >= tc.max_steps:
                break

            # ── 7a: 数据移到设备 ────────────────────────────
            x, y = x.to(device), y.to(device)

            # ── 7b: 更新学习率 ──────────────────────────────
            lr = get_lr(step, tc)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── 7c: 前向 + 反向 + 更新 ──────────────────────
            if use_amp:
                # AMP 模式（CUDA）：用 float16 加速
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)           # 前向传播
                scaler.scale(loss).backward()        # 缩放 loss 后反向传播
                scaler.unscale_(optimizer)           # 反缩放梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)  # 梯度裁剪
                scaler.step(optimizer)               # 缩放后更新权重
                scaler.update()                      # 更新 scaler 状态
            else:
                # 非 AMP 模式（CPU/MPS）：标准流程
                _, loss = model(x, y)                # 前向传播
                loss.backward()                       # 反向传播，计算梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)  # 梯度裁剪
                optimizer.step()                      # 更新权重

            # 清除梯度（set_to_none=True 省内存）
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            # ── 7d: 每 100 步打印训练 loss ──────────────────
            if step % 100 == 0:
                avg = sum(losses[-100:]) / len(losses[-100:])  # 最近 100 步平均 loss
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg:10.4f} | {'--':>10} | {elapsed:7.1f}s")

            # ── 7e: 每 eval_interval 步评估 ─────────────────
            if step > 0 and step % tc.eval_interval == 0:
                el = evaluate(model, eval_loader, device)
                avg_train = sum(losses[-tc.eval_interval:]) / min(len(losses), tc.eval_interval)
                elapsed = time.time() - t0
                print(f"{step:6d} | {lr:10.6f} | {avg_train:10.4f} | {el:10.4f} | {elapsed:7.1f}s")

                # 如果 eval loss 比之前最好的还低，保存最佳模型
                if el < best_eval:
                    best_eval = el
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": vars(mc),
                        "eval_loss": el,
                    }, os.path.join(tc.output_dir, "best_model.pt"))
                    print(f"  -> Best model (eval={el:.4f})")

            # ── 7f: 每 save_interval 步保存检查点 ───────────
            if step > 0 and step % tc.save_interval == 0:
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": vars(mc),
                }, os.path.join(tc.output_dir, f"step_{step}.pt"))

            step += 1

    # ── 第 8 步：训练结束，保存最终模型 ──────────────────────
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": vars(mc),
        "train_losses": losses,
    }, os.path.join(tc.output_dir, "final_model.pt"))

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.0f}s, best eval: {best_eval:.4f}")


if __name__ == "__main__":
    train()
