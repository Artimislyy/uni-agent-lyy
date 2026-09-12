#!/usr/bin/env python3
r"""把现有 trajectory.npz 转为 Markdown 和/或原始 TXT，不需要重新推理。

    python3 uni_agent/tasks/eda_agent/decode_trajectory.py \
        --trajectory /output/logs/EDA-rollout/某次运行 \
        --model /input/models/Qwen3.8-27B

--trajectory 可指定一个 NPZ、一个 session 或整次运行的目录（递归查找）。
默认在 NPZ 同目录生成 trajectory.md；--format both 同时输出原始 TXT。
仍支持原来的 TRAJECTORY_FILE 和 MODEL_PATH 环境变量用法。
必须使用生成轨迹时的 tokenizer；只加载本地 tokenizer，不加载模型权重。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from itertools import groupby
from pathlib import Path


def _code_block(text: str) -> str:
    # 保留轨迹自身的 Markdown/Tcl/HTML，避免其中的代码围栏破坏报告排版。
    fence = "`" * max(3, 1 + max((len(m.group()) for m in re.finditer(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}\n\n"


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _trajectory_markdown(prefix: str, prompt: list[int], response: list[int], mask: list[int], tokenizer) -> str:
    mask_valid = len(mask) == len(response) and all(value in (0, 1) for value in mask)
    parts = [
        f"## {prefix}\n\n",
        f"Prompt：{len(prompt)} token；Response：{len(response)} token；"
        f"合计：{len(prompt) + len(response)} token。\n\n",
        f"其中模型生成：{sum(mask) if mask_valid else '未知（mask 缺失或无效）'} token。\n\n",
        "<details>\n<summary>初始 Prompt（点击展开完整内容）</summary>\n\n",
        _code_block(_decode(tokenizer, prompt)),
        "</details>\n\n",
    ]
    # mask 标识真实 token 来源，不依赖模型特定的聊天模板。
    groups = groupby(range(len(response)), key=lambda i: mask[i] if mask_valid else None)
    for number, (source, indices) in enumerate(groups, start=1):
        ids = [response[i] for i in indices]
        label = {1: "模型生成", 0: "工具返回 / 后续输入", None: "后续内容（来源未知）"}[source]
        parts.append(f"### 片段 {number} · {label} · {len(ids)} token\n\n")
        parts.append(_code_block(_decode(tokenizer, ids)))
    return "".join(parts)


def decode_archive(path: Path, tokenizer, output_format: str = "md") -> list[Path]:
    import numpy as np

    outputs = []
    markdown = [
        "# Session 轨迹\n\n",
        "按模型生成和工具返回等后续输入分段，片段数不等于 Claude Code 轮数。\n\n",
        "finished=True 不代表提交或验收成功，详情请查看 task.log。\n\n",
    ]
    metadata_path = path.with_suffix(".json")
    if output_format != "txt" and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        markdown.extend(["## 统计信息\n\n", _code_block(json.dumps(metadata, ensure_ascii=False, indent=2))])
    with np.load(path, allow_pickle=False) as data:
        prefixes = [key.removesuffix("_prompt_ids") for key in data.files if key.endswith("_prompt_ids")]
        prefixes.sort(key=lambda value: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", value)])
        if not prefixes:
            raise ValueError(f"没有找到 *_prompt_ids 字段，npz 里只有：{data.files}")
        for prefix in prefixes:
            prompt = data[f"{prefix}_prompt_ids"].tolist()
            response = data[f"{prefix}_response_ids"].tolist()
            if output_format in ("md", "both"):
                mask_key = f"{prefix}_response_mask"
                mask = data[mask_key].tolist() if mask_key in data.files else []
                markdown.append(_trajectory_markdown(prefix, prompt, response, mask, tokenizer))
            if output_format in ("txt", "both"):
                output = path.parent / f"{prefix}_decoded.txt"
                output.write_text(_decode(tokenizer, prompt + response), encoding="utf-8")
                outputs.append(output)
    if output_format in ("md", "both"):
        output = path.with_suffix(".md")
        output.write_text("".join(markdown), encoding="utf-8")
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory", default=os.environ.get("TRAJECTORY_FILE"), help="NPZ 文件或日志目录")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"), help="生成该轨迹的本地模型/tokenizer 目录")
    parser.add_argument("--format", choices=("md", "txt", "both"), default="md")
    args = parser.parse_args()
    if not args.trajectory or not args.model:
        parser.error("请指定 --trajectory 和 --model，或设置 TRAJECTORY_FILE 和 MODEL_PATH")
    root = Path(args.trajectory).expanduser()
    paths = sorted(root.rglob("trajectory.npz")) if root.is_dir() else [root]
    if not paths or any(not path.is_file() for path in paths):
        parser.error(f"没有找到 trajectory.npz：{root}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model).expanduser()), local_files_only=True)
    failures = 0
    for path in paths:
        try:
            for output in decode_archive(path, tokenizer, args.format):
                print(f"已保存：{output}")
        except Exception as exc:
            failures += 1
            print(f"转换失败：{path}: {exc}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
