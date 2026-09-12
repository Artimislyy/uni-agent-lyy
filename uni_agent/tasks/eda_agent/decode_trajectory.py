#!/usr/bin/env python3
"""把一条 session 的 trajectory.npz 解码成可读文本。

用法：
    export TRAJECTORY_FILE="/实际日志目录/session-xxx/trajectory.npz"
    export MODEL_PATH="/实际模型目录/Qwen3.5-4B"
    python3 decode_trajectory.py

对 npz 里的每条轨迹（*_prompt_ids / *_response_ids），解码出完整对话文本
（保留特殊标记，便于区分用户 / 模型 / 工具消息），保存到 npz 同目录下的
<前缀>_decoded.txt。
"""

import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main() -> None:
    path = Path(os.environ["TRAJECTORY_FILE"])
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["MODEL_PATH"],
        local_files_only=True,
    )

    with np.load(path, allow_pickle=False) as data:
        prompt_keys = sorted(key for key in data.files if key.endswith("_prompt_ids"))
        if not prompt_keys:
            raise SystemExit(f"没有找到 *_prompt_ids 字段，npz 里只有：{data.files}")

        for key in prompt_keys:
            prefix = key.removesuffix("_prompt_ids")
            prompt_ids = data[key].tolist()
            response_ids = data[f"{prefix}_response_ids"].tolist()

            text = tokenizer.decode(
                prompt_ids + response_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            output = path.parent / f"{prefix}_decoded.txt"
            output.write_text(text, encoding="utf-8")
            print(f"已保存：{output}")
            print(f"prompt={len(prompt_ids)}, response={len(response_ids)}")


if __name__ == "__main__":
    main()
