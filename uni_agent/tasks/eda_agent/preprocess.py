#!/usr/bin/env python3
"""将 Innovus repair 数据转换为 Uni-Agent 训练数据。

处理流程：

1. 从 ``tasks/index.tsv`` 读取权威任务清单。
2. 用 ``task.json`` 和 ``metadata.json`` 校验任务信息。
3. 按数据来源划分训练集和验证集，并检查同源数据泄漏。
4. 将 system prompt、SKILL.md 和 user prompt 组合成提示词，task.md 随任务文件上传沙箱。
5. 输出 JSON/Parquet、划分清单和审计信息。

示例：

    python3 -m uni_agent.tasks.eda_agent.preprocess \
        --dataset-root DATASET \
        --output-dir OUTPUT \
        --output-format json

使用 ``--check-only`` 可以只检查数据和划分，不生成文件。
使用 ``--smoke-test`` 按 task_id 顺序选取 4 条训练数据和 2 条验证数据，用于打通训练流程。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

# ============================== 基本配置 ==============================

DATA_SOURCE = "dataset_innovus_19_10"
TASK_NAME = "eda"
DATASET_ROOT_ENV = "EDA_DATASET_ROOT"
OUTPUT_PREFIX = "eda"
READY_STATUSES = {"PASS", "PASS_FRESH_LOAD_GOLDEN"}

# agent 运行前只能上传 VISIBLE_PATHS；答案和验证器必须等 agent 结束后再上传。
VISIBLE_PATHS = ["task.md", "initial_state/design.enc", "initial_state/design.enc.dat"]
HIDDEN_PATHS = ["verifier", "reference", "initial_state/reports"]
VERIFIER_PATH = "verifier/verify.tcl"
ANSWER_PATH = "repair.tcl"
RESULT_PATH = "verifier_output/result.json"

# 固定按数据来源划分，避免同源或近重复任务同时出现在训练集和验证集。
VALIDATION_SOURCES = {
    "BPFE-6A_hold_coupled_v0",
    "SWERV-2A_autoopt_hard_v0",
    "SWERV-3A_HOLD_data_v0",
    "jpeg_easy_medium_v3",
    "jpeg_hold_coupled_v0",
}

# manifest 保存每个任务的来源和划分，便于人工复查。
MANIFEST_FIELDS = (
    "split",
    "task_id",
    "task_relpath",
    "design",
    "task_type",
    "difficulty",
    "source_dataset",
    "source_sample",
    "validation_status",
    "task_md_sha256",
)


class PreprocessError(RuntimeError):
    """表示输入数据或命令参数不符合预期。"""

    pass


@dataclass(frozen=True)
class Sample:
    """一个已经校验并标准化的 Innovus 任务。"""

    task_id: str
    root: Path
    relpath: str
    task_md: str
    task_json: dict[str, Any]
    metadata_json: dict[str, Any]
    design: str
    task_type: str
    difficulty: str
    repair_family: str
    source_dataset: str
    source_sample: str
    validation_status: str
    tool: str
    tool_version: str
    task_md_sha256: str


# ============================== 读取与校验 ==============================


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON，并保证顶层结构是字典。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessError(f"Failed to read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreprocessError(f"The top-level value in {path} must be a JSON object")
    return value


def as_dict(value: Any) -> dict[str, Any]:
    """把非字典值安全地转换为空字典，兼容不同版本的 metadata。"""

    return value if isinstance(value, dict) else {}


def as_text(value: Any, default: str = "") -> str:
    """把 metadata 中可能出现的字符串或复合值统一转成文本。"""

    if value is None:
        return default
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_text(value: Any) -> str:
    """异构字段统一存成字符串，避免生成 Parquet 时出现 schema 冲突。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    """计算文件哈希，用于判断源任务或输出文件是否发生变化。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_path(task_root: Path, relative_path: str, *, directory: bool = False) -> Path:
    """确认任务资源存在，并拒绝访问任务目录之外的路径。"""

    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreprocessError(f"Unsafe task path: {relative_path}")
    path = task_root / relative
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise PreprocessError(f"Missing task asset: {path}")
    return path


def read_index(tasks_root: Path) -> tuple[list[dict[str, str]], Path]:
    """读取权威索引，检查必需字段和重复 task_id。"""

    index_path = tasks_root / "index.tsv"
    if not index_path.is_file():
        raise PreprocessError(f"Missing task index: {index_path}")

    with index_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"task_id", "task_type", "source_dataset", "source_sample", "validation_status"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise PreprocessError(f"index.tsv is missing columns: {sorted(missing)}")
        rows = list(reader)

    task_ids = [row["task_id"] for row in rows]
    duplicates = [task_id for task_id, count in Counter(task_ids).items() if count > 1]
    if duplicates:
        raise PreprocessError(f"index.tsv contains duplicate task IDs: {duplicates}")
    return rows, index_path


def get_difficulty(task_json: dict[str, Any], metadata_json: dict[str, Any]) -> tuple[str, str]:
    """兼容不同任务来源的字段布局，提取难度和 repair family。"""

    metadata_difficulty = as_dict(metadata_json.get("difficulty"))
    task_difficulty = task_json.get("difficulty")
    if isinstance(task_difficulty, dict):
        task_difficulty = task_difficulty.get("label")
    difficulty = as_text(metadata_difficulty.get("label") or task_difficulty, "unspecified")
    repair_family = as_text(
        metadata_difficulty.get("repair_family_id") or task_json.get("benchmark_subtype"),
        "unspecified",
    )
    return difficulty, repair_family


def load_sample(dataset_root: Path, task_root: Path, index_row: dict[str, str]) -> Sample:
    """读取一个任务，交叉校验三个元数据来源并生成标准 Sample。"""

    task_id = index_row["task_id"]
    task_md_path = require_path(task_root, "task.md")
    task_json = read_json(require_path(task_root, "task.json"))
    metadata_json = read_json(require_path(task_root, "metadata.json"))

    if as_text(task_json.get("task_id"), task_id) != task_id:
        raise PreprocessError(f"{task_id}: task_id does not match index.tsv")
    if as_text(task_json.get("task_type")) != index_row["task_type"]:
        raise PreprocessError(f"{task_id}: task_type does not match index.tsv")

    packaged_status = as_text(as_dict(metadata_json.get("packaged_validation")).get("status"))
    if packaged_status != index_row["validation_status"]:
        raise PreprocessError(f"{task_id}: validation_status does not match index.tsv")

    source = as_dict(metadata_json.get("source"))
    source_dataset = as_text(source.get("dataset") or index_row["source_dataset"])
    source_sample = as_text(source.get("sample_id") or index_row["source_sample"])
    if (source_dataset, source_sample) != (index_row["source_dataset"], index_row["source_sample"]):
        raise PreprocessError(f"{task_id}: metadata.source does not match index.tsv")

    # 模型只能看到 task.md 和 checkpoint，不能暴露 task.json、报告或答案。
    for path in VISIBLE_PATHS:
        require_path(task_root, path, directory=path.endswith(".dat"))
    require_path(task_root, VERIFIER_PATH)
    require_path(task_root, "reference", directory=True)

    task_md = task_md_path.read_text(encoding="utf-8-sig").strip()
    if not task_md:
        raise PreprocessError(f"{task_id}: task.md is empty")
    difficulty, repair_family = get_difficulty(task_json, metadata_json)
    environment = as_dict(task_json.get("environment"))
    tool_metadata = as_dict(metadata_json.get("tool"))
    return Sample(
        task_id=task_id,
        root=task_root,
        relpath=task_root.relative_to(dataset_root).as_posix(),
        task_md=task_md,
        task_json=task_json,
        metadata_json=metadata_json,
        design=as_text(task_json.get("design") or task_root.parent.name),
        task_type=index_row["task_type"],
        difficulty=difficulty,
        repair_family=repair_family,
        source_dataset=source_dataset,
        source_sample=source_sample,
        validation_status=index_row["validation_status"],
        tool=as_text(environment.get("tool") or tool_metadata.get("name")),
        tool_version=as_text(environment.get("version") or tool_metadata.get("version")),
        task_md_sha256=sha256_file(task_md_path),
    )


def load_samples(dataset_root: Path) -> tuple[list[Sample], list[dict[str, str]], str]:
    """按 index.tsv 加载全部可用任务，同时记录被过滤的任务。"""

    tasks_root = dataset_root / "tasks"
    index_rows, index_path = read_index(tasks_root)

    # index.tsv 是权威清单，目录扫描只负责定位任务。
    task_paths = list(tasks_root.glob("*/*/task.md"))
    directory_ids = [path.parent.name for path in task_paths]
    duplicate_ids = [task_id for task_id, count in Counter(directory_ids).items() if count > 1]
    if duplicate_ids:
        raise PreprocessError(f"Multiple task directories use the same task ID: {sorted(duplicate_ids)}")

    task_dirs = {path.parent.name: path.parent for path in task_paths}
    index_ids = {row["task_id"] for row in index_rows}
    if set(task_dirs) != index_ids:
        raise PreprocessError(
            f"Task directories do not match index.tsv: missing={sorted(index_ids - set(task_dirs))}, "
            f"extra={sorted(set(task_dirs) - index_ids)}"
        )

    samples: list[Sample] = []
    excluded: list[dict[str, str]] = []
    for row in index_rows:
        if row["validation_status"] not in READY_STATUSES:
            excluded.append(
                {
                    "task_id": row["task_id"],
                    "validation_status": row["validation_status"],
                    "reason": "validation_status_not_ready",
                }
            )
            continue
        samples.append(load_sample(dataset_root, task_dirs[row["task_id"]], row))
    return sorted(samples, key=lambda sample: sample.task_id), excluded, sha256_file(index_path)


# ============================== 数据集划分 ==============================


def leakage_keys(sample: Sample) -> set[str]:
    """提取明确同源或高度近似的任务标识。"""

    source = as_dict(sample.metadata_json.get("source"))
    difficulty = as_dict(sample.metadata_json.get("difficulty"))

    # 同一个来源样本属于明确的数据泄露；仅来源数据集相同不代表是同一个任务。
    keys = {
        f"sample:{sample.source_dataset}/{sample.source_sample}",
    }
    values = {
        "source_state_signature": source.get("state_signature"),  # 原始设计状态签名
        "difficulty_state_signature": difficulty.get("state_signature"),  # 派生状态签名
        "transitive_group": sample.metadata_json.get(
            "transitive_composition_group_id"
        ),  # 组合任务分组
    }
    keys.update(f"{name}:{as_text(value)}" for name, value in values.items() if value)

    # repair_family 只是修复类型，不是同源标识；terminal cone 需结合设计名称判断。
    terminal_cone = as_text(difficulty.get("terminal_cone_id"))
    if terminal_cone:
        keys.add(f"terminal_cone:{sample.design}/{terminal_cone}")

    # 组合任务及其历史来源样本不能分布在训练集和验证集两侧。
    lineage = sample.metadata_json.get("transitive_lineage")
    if isinstance(lineage, list):
        for item in lineage:
            if isinstance(item, dict):
                dataset = as_text(item.get("dataset"))
                case_id = as_text(item.get("case_id") or item.get("sample_id"))
                if dataset and case_id:
                    keys.add(f"sample:{dataset}/{case_id}")
    return keys


def split_samples(samples: list[Sample]) -> dict[str, list[Sample]]:
    """按配置的来源划分数据，并确认两侧没有同源标识。"""

    splits = {
        "train": [sample for sample in samples if sample.source_dataset not in VALIDATION_SOURCES],
        "validation": [sample for sample in samples if sample.source_dataset in VALIDATION_SOURCES],
    }
    if not splits["train"] or not splits["validation"]:
        raise PreprocessError("Both train and validation splits must contain at least one sample")

    train_keys = set().union(*(leakage_keys(sample) for sample in splits["train"]))
    validation_keys = set().union(*(leakage_keys(sample) for sample in splits["validation"]))
    overlap = sorted(train_keys & validation_keys)
    if overlap:
        raise PreprocessError(f"Train/validation leakage detected: {overlap[:10]}")

    return splits


# ============================== 构造训练样本 ==============================


def make_prompt(system_prompt: str, user_prompt: str, skill_prompt: str) -> list[dict[str, str]]:
    """构造 ClaudeCodeAgent 所需的单条 user 消息。"""

    # 保留单条 user 消息，把操作规则、技能全文和任务请求合并；task.md 由 Agent 在沙箱读取。
    content = "\n\n".join(
        [
            f"## EDA agent operating instructions\n\n{system_prompt.strip()}",
            f"## Innovus ECO closure skill (SKILL.md)\n\n{skill_prompt.strip()}",
            f"## Task request\n\n{user_prompt.strip()}",
        ]
    )
    return [{"role": "user", "content": content}]


def make_row(
    sample: Sample,
    split: str,
    system_prompt: str,
    user_prompt: str,
    skill_prompt: str,
    dataset_root_env: str,
) -> dict[str, Any]:
    """把 Sample 转换成 Uni-Agent 读取的一行数据。"""

    metadata = {
        "task_id": sample.task_id, #题目的编号，task_0001
        "task_relpath": sample.relpath.removeprefix("tasks/"),#ibex_top/task_0001
        "design": sample.design, #电路设计名称，ibex_top、jpeg_encoder
        "task_type": sample.task_type, #需要修复的问题类型，setup_repair 表示修复 setup 时序问题
        "difficulty": sample.difficulty, #题目难度，easy、medium、hard；缺少时可能为 unspecified
        "repair_family_id": sample.repair_family, #修复类型或方法类别的标识；同类修复不一定来自同一道原题
        "source_dataset": sample.source_dataset,#哪个来源数据集，jpeg_easy_medium_v3
        "source_sample": sample.source_sample, # 来源数据集中的样本编号，jpeg_easy_medium_v3/0001
        "validation_status": sample.validation_status, #PASS、PASS_FRESH_LOAD_GOLDEN 等，表示任务是否通过验证
        "split": split,#train 或 validation，表示该任务属于训练集还是验证集
        "tool": sample.tool,#EDA 工具名称，通常是 Innovus
        "tool_version": sample.tool_version, #EDA 工具版本号，通常是 Innovus 的版本号
        "objective_json": json_text(sample.task_json.get("objective", {})), #从 task.json 提取的修复目标，转换成 JSON 字符串保存,需要达到的时序、DRV 等要求
        "initial_metrics_json": json_text(sample.task_json.get("initial_metrics", {})),#从 task.json 提取的初始设计指标,修复前的 WNS、TNS、违例数量
        "forbidden_commands_json": json_text(sample.task_json.get("forbidden_commands", [])),#从 task.json 提取的禁用命令列表, '["optDesign"]'
    }
    return {
        "schema_version": 1,
        "data_source": DATA_SOURCE, #dataset_innovus_19_10
        "instance_id": sample.relpath.removeprefix("tasks/"), #ibex_top/task_0001
        "prompt": make_prompt(system_prompt, user_prompt, skill_prompt),
        "extra_info": {
            "tools_kwargs": {
                "task": {
                    "name": TASK_NAME, #eda
                    "task_root": sample.relpath, #ibex_top/task_0001
                    "dataset_root_env": dataset_root_env, #EDA_DATASET_ROOT
                    "visible_paths": VISIBLE_PATHS, #["task.md", "initial_state/design.enc", "initial_state/design.enc.dat"]
                    "hidden_paths": [path for path in HIDDEN_PATHS if (sample.root / path).exists()],
                    "verifier_path": VERIFIER_PATH,#"verifier/verify.tcl"
                    "answer_path": ANSWER_PATH,#"repair.tcl"
                    "result_path": RESULT_PATH,#"verifier_output/result.json"
                    "metadata": metadata, #任务元数据
                }
            }
        },
    }


def make_manifest(splits: dict[str, list[Sample]]) -> list[dict[str, str]]:
    """生成任务到 train/validation 的可读映射。"""

    rows = []
    for split, samples in splits.items():
        for sample in samples:
            rows.append(
                {
                    "split": split,
                    "task_id": sample.task_id,
                    "task_relpath": sample.relpath,
                    "design": sample.design,
                    "task_type": sample.task_type,
                    "difficulty": sample.difficulty,
                    "source_dataset": sample.source_dataset,
                    "source_sample": sample.source_sample,
                    "validation_status": sample.validation_status,
                    "task_md_sha256": sample.task_md_sha256,
                }
            )
    return rows


def count_lfs_pointers(samples: list[Sample]) -> int:
    """统计可见资源中尚未下载的 Git LFS 指针文件。"""

    count = 0
    for sample in samples:
        paths = [sample.root / path for path in VISIBLE_PATHS[:2]]
        paths.extend(path for path in (sample.root / VISIBLE_PATHS[2]).rglob("*") if path.is_file())
        for path in paths:
            with path.open("rb") as file:
                count += file.read(128).startswith(b"version https://git-lfs.github.com/spec/v1\n")
    return count


def print_summary(splits: dict[str, list[Sample]], lfs_count: int, preview: int) -> None:
    """打印划分数量和主要分布，供运行者快速检查。"""

    print("Split strategy: family-holdout-v1")
    for split, samples in splits.items():
        designs = dict(sorted(Counter(sample.design for sample in samples).items()))
        task_types = dict(sorted(Counter(sample.task_type for sample in samples).items()))
        print(f"  {split}: {len(samples)}")
        print(f"    design: {designs}")
        print(f"    task_type: {task_types}")
        if preview:
            print(f"    preview: {[sample.task_id for sample in samples[:preview]]}")
    print(f"Git LFS pointers: {lfs_count}")


# ============================== 写入输出 ==============================


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入带缩进的 JSON 数组，并核对任务数量。"""

    content = json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")
    if len(json.loads(content)) != len(rows):
        raise PreprocessError(f"JSON row count mismatch: {path}")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入训练使用的 Parquet，并重新读取以核对行数。"""

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path)
    if pq.read_table(path).num_rows != len(rows):
        raise PreprocessError(f"Parquet row count mismatch: {path}")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """写入制表符分隔的划分清单。"""

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    splits: dict[str, list[Sample]],
    system_prompt: str,
    user_prompt: str,
    skill_prompt: str,
    args: argparse.Namespace,
    excluded: list[dict[str, str]],
    index_sha256: str,
    lfs_count: int,
    eligible_count: int,
) -> None:
    """生成数据文件、manifest 和 audit；默认不覆盖已有文件。"""

    extensions = ("json", "parquet") if args.output_format == "both" else (args.output_format,)
    data_paths = {
        (split, extension): output_dir / f"{OUTPUT_PREFIX}_{split}.{extension}"
        for split in splits
        for extension in extensions
    }
    manifest_path = output_dir / f"{OUTPUT_PREFIX}_split_manifest.tsv"
    audit_path = output_dir / f"{OUTPUT_PREFIX}_audit.json"
    output_paths = [manifest_path, audit_path, *data_paths.values()]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise PreprocessError(f"Output files already exist; use --overwrite: {existing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {
        split: [
            make_row(sample, split, system_prompt, user_prompt, skill_prompt, args.dataset_root_env)
            for sample in samples
        ]
        for split, samples in splits.items()
    }
    write_manifest(manifest_path, make_manifest(splits))
    for (split, extension), path in data_paths.items():
        writer = write_json if extension == "json" else write_parquet
        writer(path, rows_by_split[split])

    audit = {
        "source": {"dataset_root": str(args.dataset_root.resolve()), "index_sha256": index_sha256},
        "selection": {
            "eligible": eligible_count,
            "selected": sum(map(len, splits.values())),
            "smoke_test": args.smoke_test,
            "excluded": excluded,
        },
        "split": {
            "strategy": "family-holdout-v1",
            "counts": {name: len(samples) for name, samples in splits.items()},
            "validation_sources": sorted(VALIDATION_SOURCES),
        },
        "checks": {"task_ids_disjoint": True, "leakage_keys_disjoint": True},
        "visibility": {"visible": VISIBLE_PATHS, "hidden": HIDDEN_PATHS},
        "git_lfs_pointer_count": lfs_count,
        "outputs": {path.name: sha256_file(path) for path in [manifest_path, *data_paths.values()]},
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Output written to: {output_dir}")


# ============================== 命令入口 ==============================


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Build Uni-Agent data from Innovus repair tasks."
    )  # 创建命令行参数解析器，并设置帮助信息中的程序说明。
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing tasks/index.tsv.",
    )  # 原始数据集根目录，目录下必须包含 tasks/index.tsv；这是必填参数。
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory used to save generated dataset files.",
    )  # 生成文件的保存目录；使用 --check-only 时可以不填写。
    parser.add_argument(
        "--dataset-root-env",
        default=DATASET_ROOT_ENV,
        help=f"Runtime environment variable for the dataset root; default: {DATASET_ROOT_ENV}.",
    )  # 运行任务时用于定位数据集根目录的环境变量名。
    parser.add_argument(
        "--output-format",
        choices=("json", "parquet", "both"),
        default="both",
        help="Output format; default: both.",
    )  # 输出格式：JSON、Parquet，或者同时生成两种格式；默认同时生成。
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Select the first 4 training and 2 validation samples by task_id for a training smoke test.",
    )  # 保持原有来源划分，只输出用于打通训练流程的小数据集。
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Preview the first N task IDs of each split; default: 5.",
    )  # 在终端中预览每个数据集的前几个 task_id；默认显示 5 个。
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the dataset and split without writing output files.",
    )  # 只检查数据与划分结果，不生成任何输出文件。
    parser.add_argument(
        "--require-lfs-materialized",
        action="store_true",
        help="Fail in check-only mode if any Git LFS pointer files remain.",
    )  # 在检查模式下也要求 LFS 文件已经下载；发现指针文件时立即报错。
    parser.add_argument(
        "--allow-lfs-pointers",
        action="store_true",
        help="Allow output files to reference unresolved Git LFS pointers.",
    )  # 明确允许生成仍引用 Git LFS 指针的输出；这些输出通常不能直接运行。
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )  # 允许覆盖输出目录中已经存在的同名文件。
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """串联读取、划分、检查和写入流程。"""

    args = build_parser().parse_args(argv)
    try:
        if args.preview < 0:
            raise PreprocessError("--preview must be non-negative")
        if not args.dataset_root_env.strip():
            raise PreprocessError("--dataset-root-env must not be empty")
        dataset_root = args.dataset_root.expanduser().resolve()
        args.dataset_root = dataset_root

        # 固定读取数据集同级 prompts 下的三个文件，读一次后供 make_prompt 拼接。
        prompt_root = dataset_root.parent / "prompts"
        system_path = prompt_root / "system_prompt.txt"
        user_path = prompt_root / "user_prompt.txt"
        skill_path = prompt_root / "innovus-eco-closure-mcp/SKILL.md"
        system_prompt = system_path.read_text(encoding="utf-8-sig").strip()
        user_prompt = user_path.read_text(encoding="utf-8-sig").strip()
        skill_prompt = skill_path.read_text(encoding="utf-8-sig").strip()
        if not system_prompt or not user_prompt or not skill_prompt:
            raise PreprocessError("system_prompt.txt, user_prompt.txt and SKILL.md must not be empty")

        samples, excluded, index_sha256 = load_samples(dataset_root)
        splits = split_samples(samples)
        if args.smoke_test:
            limits = {"train": 4, "validation": 2}
            for split, limit in limits.items():
                if len(splits[split]) < limit:
                    raise PreprocessError(
                        f"--smoke-test requires at least {limit} {split} samples; found {len(splits[split])}"
                    )
            # load_samples 已按 task_id 排序，固定取前几条使试跑数据可复现。
            splits = {split: group[:limits[split]] for split, group in splits.items()}
        selected_samples = [sample for group in splits.values() for sample in group]
        lfs_count = count_lfs_pointers(selected_samples)
        print_summary(splits, lfs_count, args.preview)

        # 检查模式默认只报告 LFS 状态；正式生成时默认要求真实资源已经下载。
        lfs_required = args.require_lfs_materialized or (not args.check_only and not args.allow_lfs_pointers)
        if lfs_required and lfs_count:
            raise PreprocessError(f"Found {lfs_count} Git LFS pointers; run git lfs pull first")
        if args.check_only:
            return 0
        if args.output_dir is None:
            raise PreprocessError("--output-dir is required unless --check-only is used")

        write_outputs(
            args.output_dir.expanduser().resolve(),
            splits,
            system_prompt,
            user_prompt,
            skill_prompt,
            args,
            excluded,
            index_sha256,
            lfs_count,
            len(samples),
        )
        return 0
    except (OSError, PreprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
