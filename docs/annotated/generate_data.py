"""GuppyLM 数据生成模块 — 60 个主题的模板对话生成器。

这个文件约 1700 行，核心原理很简单：模板组合。

原理：
    每个主题定义一组"用户消息模板"和"Guppy 回复模板"，
    模板中包含随机变量（从词汇池中抽取），
    每次调用随机组合，产生独特的对话。

    例如：
        用户模板: "are you hungry", "want some food", "time to eat", ...
        Guppy 模板: "yes. always yes. give me the {FOOD}.",
                    "i could eat. i like the {FOOD} best.", ...

    随机变量：
        FOOD = pick(FOOD_TYPES)  → "flakes" / "pellets" / "bloodworms" / ...
        SPOT = pick(TANK_SPOTS)  → "near the rock" / "behind the plant" / ...

    30 个鱼缸物品 × 17 种食物 × 25 种活动 × 60 个模板 ≈ 16K 种独特回复
    生成 60K 条数据时，大部分样本都是唯一的。

60 个主题覆盖：
    问候、感受、温度（冷/热）、食物、光线、水、自我介绍、
    困惑（不理解人类事物）、鱼缸、噪音、夜晚、孤独、
    气泡、玻璃、倒影、呼吸、游泳、颜色、味道、植物、
    过滤器、藻类、蜗牛、玻璃敲击、害怕、兴奋、无聊、
    好奇、快乐、疲惫、外面、猫、雨、季节、音乐、访客、
    孩子、生命意义、时间、记忆、梦想、大小、未来、过去、
    名字、天气、睡觉、朋友、笑话、恐惧、爱、年龄、智力、
    健康、唱歌、电视

使用方式：
    from guppylm.generate_data import generate_dataset
    generate_dataset(n_samples=60000, eval_ratio=0.05)

    生成后写入 data/train.jsonl 和 data/eval.jsonl
"""

import json
import random
import os
from collections import Counter

random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
#  基础工具函数
# ══════════════════════════════════════════════════════════════════════════════

def pick(lst):
    """从列表中随机选一个元素。"""
    return random.choice(lst)


def pick_n(lst, n):
    """从列表中随机选 n 个不重复元素。"""
    return random.sample(lst, min(n, len(lst)))


def maybe(text, p=0.5):
    """以概率 p 返回文本，否则返回空字符串。
    用于生成可选的附加句子，增加回复的多样性。
    """
    return text if random.random() < p else ""


def join_sentences(*parts):
    """将多个句子片段拼接为完整句子。
    自动过滤空字符串，清理多余空格。
    """
    return " ".join(p.strip() for p in parts if p.strip()).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  词汇池：Guppy 世界的所有"词汇"
# ══════════════════════════════════════════════════════════════════════════════

# 鱼缸里的物品（30 种）
TANK_OBJECTS = [
    "rock", "big rock", "small rock", "pebble", "plant", "fake plant",
    "castle", "cave", "log", "driftwood", "shell", "coral piece",
    "moss ball", "ceramic pot", "bridge", "tunnel", "arch", "skull decoration",
    "treasure chest", "anchor", "shipwreck", "bubble wall", "thermometer",
    "heater tube", "filter tube", "glass wall", "gravel", "sand",
]

# Guppy 喜欢待的位置（20 种）
TANK_SPOTS = [
    "near the rock", "behind the plant", "by the filter", "in the corner",
    "at the top", "near the bottom", "by the glass", "under the log",
    "next to the cave", "near the heater", "by the bubbles", "in the middle",
    "behind the castle", "near the gravel", "along the glass wall",
    "between the rocks", "under the bridge", "in my favorite spot",
    "where the current is gentle", "where the light hits the gravel",
]

# 食物类型（17 种）
FOOD_TYPES = [
    "flakes", "pellets", "orange flakes", "tiny pellets", "the green ones",
    "the red flakes", "bloodworms", "brine shrimp", "daphnia", "tubifex",
    "the crunchy ones", "the soft ones", "the sinking ones", "the floating ones",
    "algae wafer", "micro pellets", "freeze dried worms",
]

# 水的描述（16 种）
WATER_DESCRIPTIONS = [
    "clear", "fresh", "cool", "warm", "just right", "a little cloudy",
    "very clean", "slightly different", "normal", "perfect", "new",
    "crisp", "gentle", "calm", "bubbly", "still",
]

# Guppy 的活动（25 种）
ACTIVITIES = [
    "swimming in circles", "looking at the glass", "following a bubble",
    "hiding behind the rock", "resting near the bottom", "hovering",
    "investigating a speck", "staring at the plant", "opening and closing my mouth",
    "doing laps", "chasing my tail", "watching the bubbles go up",
    "nudging a pebble", "floating near the top", "exploring the cave",
    "pretending to be a rock", "practicing my turns", "swimming backwards badly",
    "trying to eat a bubble", "racing the current", "sitting on the gravel",
    "pressing my face against the glass", "blowing tiny bubbles",
    "wiggling my fins", "thinking about food",
]

# Guppy 的感受（15 种）
FEELINGS = [
    "good", "ok", "fine", "content", "calm", "a little hungry",
    "pretty good", "normal", "peaceful", "relaxed", "happy",
    "a bit sleepy", "curious", "comfortable", "not bad",
]

# 人类抽象概念（Guppy 不理解的，30 种）
HUMAN_THINGS = [
    "politics", "money", "the internet", "email", "taxes", "a phone",
    "driving", "a movie", "school", "work", "a computer", "math",
    "reading", "the news", "social media", "cooking", "shopping",
    "a job", "rent", "the stock market", "a meeting", "homework",
    "an app", "a password", "wifi", "bluetooth", "a podcast",
    "cryptocurrency", "a spreadsheet", "an alarm clock", "a car",
]

# ... 还有其他词汇池：WATER_THINGS, LIGHT_STATES, TIMES_OF_DAY,
# BODY_PARTS, SOUNDS 等，见原始文件


# ══════════════════════════════════════════════════════════════════════════════
#  模板生成器：每个主题一个函数
# ══════════════════════════════════════════════════════════════════════════════

def _guppy_greeting():
    """问候主题的 Guppy 回复生成器。

    结构：开场白 + 中间句 + 可选附加句
    每个部分从各自的模板列表中随机选择。
    """
    openers = [
        "hello.", "hi.", "oh hello.", "oh hi.", "hey.", "hi there.",
    ]
    middles = [
        f"i was just {pick(ACTIVITIES)}.",
        f"the water is {pick(WATER_DESCRIPTIONS)} today.",
        f"i'm {pick(TANK_SPOTS)}.",
        f"i didn't see you there. my eyes are on the sides.",
        f"are you the big shape that feeds me.",
        # ... 更多模板
    ]
    extras = [
        f"i blew some bubbles earlier.",
        f"the {pick(WATER_THINGS)} feels nice.",
        "it's a good day to be a fish.",
        "",  # 空字符串 = 不添加附加句
        "",
        "",
    ]
    return join_sentences(pick(openers), pick(middles), pick(extras))


def _guppy_food():
    """食物主题的 Guppy 回复生成器。"""
    starters = [
        "yes. always yes.",
        "food. did you say food.",
        "i could eat. i can always eat.",
        # ...
    ]
    middles = [
        f"give me the {pick(FOOD_TYPES)}.",
        f"i like the {pick(FOOD_TYPES)} best.",
        "i will swim to the top right now.",
        # ...
    ]
    extras = [
        "please.",
        "i promise to eat all of it.",
        "",
        "",
    ]
    return join_sentences(pick(starters), pick(middles), pick(extras))


def _guppy_confused(thing=None):
    """困惑主题：Guppy 不理解人类事物时的回复。

    Args:
        thing: 人类概念，如 "politics"、"money"。None 时随机选一个。
    """
    if thing is None:
        thing = pick(HUMAN_THINGS)
    starters = [
        f"i don't know what {thing} is.",
        f"{thing}. that sounds like a human thing.",
        f"is {thing} something that lives in water.",
        # ...
    ]
    deflections = [
        "is it wet.",
        "i am a fish.",
        "can you explain it in terms of water or food.",
        "my brain is the size of a seed.",
        f"can we talk about {pick(FOOD_TYPES)} instead.",
        # ...
    ]
    return join_sentences(pick(starters), pick(deflections))


# ... 还有 57 个类似的生成器函数，覆盖所有主题


# ══════════════════════════════════════════════════════════════════════════════
#  数据生成主函数
# ══════════════════════════════════════════════════════════════════════════════

# 所有生成器的注册表：(用户消息生成器, Guppy 回复生成器, 主题名)
GENERATORS = [
    (_user_greeting, _guppy_greeting, "greeting"),
    (_user_food, _guppy_food, "food"),
    (_user_confused, _guppy_confused, "confused"),
    # ... 60 个主题
]


def generate_dataset(n_samples=60000, eval_ratio=0.05, output_dir="data"):
    """生成完整的训练数据集。

    流程：
    1. 从 GENERATORS 中随机选择主题
    2. 调用用户消息生成器 → 得到 input
    3. 调用 Guppy 回复生成器 → 得到 output
    4. 格式化为 ChatML 格式
    5. 按 eval_ratio 分割为训练集和验证集
    6. 保存为 JSONL 文件

    Args:
        n_samples: 总样本数
        eval_ratio: 验证集比例（0.05 = 5%）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    samples = []
    for _ in range(n_samples):
        # 随机选择一个主题
        user_fn, guppy_fn, category = pick(GENERATORS)

        # 生成对话对
        user_msg = user_fn()
        guppy_msg = guppy_fn()

        # 格式化为 ChatML
        text = (
            f"<think>user\n{user_msg}</think>\n"
            f"<think>assistant\n{guppy_msg}</think>"
        )

        samples.append({"text": text, "category": category})

    # 分割为训练集和验证集
    n_eval = int(n_samples * eval_ratio)
    n_train = n_samples - n_eval
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    # 保存
    for name, data in [("train.jsonl", train_samples), ("eval.jsonl", eval_samples)]:
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")
        print(f"  {path}: {len(data):,} samples")

    # 打印主题分布统计
    categories = Counter(s["category"] for s in samples)
    print(f"\nCategory distribution ({len(categories)} topics):")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")
