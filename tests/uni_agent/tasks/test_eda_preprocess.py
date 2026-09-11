from __future__ import annotations

import csv
import json

import pytest

from uni_agent.tasks.eda_agent import preprocess

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def dataset(tmp_path, request):
    """构造小数据集，覆盖真实读取、划分和输出流程，无需 Innovus。"""

    root = tmp_path / "dataset_innovus_19_10"
    prompt_root = tmp_path / "prompts"
    prompts = {
        "system": prompt_root / "system_prompt.txt",
        "user": prompt_root / "user_prompt.txt",
        "skill": prompt_root / "innovus-eco-closure-mcp/SKILL.md",
    }
    for name, path in prompts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"\ufeff {name} instructions \n", encoding="utf-8")

    index = ["task_id\ttask_type\tsource_dataset\tsource_sample\tvalidation_status"]
    sources = getattr(request, "param", ["training_source", "jpeg_easy_medium_v3"])
    for number, source in enumerate(sources, start=1):
        task_id = f"task_{number:04d}"
        task = root / "tasks/design" / task_id
        files = {
            "task.md": f"Repair {task_id}",
            "task.json": json.dumps({"task_id": task_id, "task_type": "setup_repair"}),
            "metadata.json": json.dumps({"packaged_validation": {"status": "PASS"}}),
            "initial_state/design.enc": "# checkpoint loader",
            "initial_state/design.enc.dat/db.bin": "checkpoint data",
            "verifier/verify.tcl": "# verifier",
            "reference/repair.tcl": "# reference",
        }
        for relative, text in files.items():
            path = task / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        index.append(f"{task_id}\tsetup_repair\t{source}\t{task_id}\tPASS")
    (root / "tasks/index.tsv").write_text("\n".join(index) + "\n", encoding="utf-8")
    return root, prompts


def test_make_prompt_keeps_full_skill_in_one_user_message():
    skill = "---\nname: eco\n---\n# Skill\n```tcl\nputs done\n```\nEnd of skill"
    prompt = preprocess.make_prompt(" system ", " user ", skill)

    expected_content = "\n\n".join([
        "## EDA agent operating instructions\n\nsystem",
        f"## Innovus ECO closure skill (SKILL.md)\n\n{skill}",
        "## Task request\n\nuser",
    ])
    assert prompt == [{"role": "user", "content": expected_content}]


@pytest.mark.parametrize("output_format", ["json", "parquet"])
def test_default_prompts_reach_generated_rows(dataset, tmp_path, output_format):
    if output_format == "parquet":
        pq = pytest.importorskip("pyarrow.parquet")
    root, _ = dataset
    output = tmp_path / "output"

    assert preprocess.main([
        "--dataset-root", str(root), "--output-dir", str(output),
        "--output-format", output_format, "--preview", "0",
    ]) == 0

    for split in ("train", "validation"):
        path = output / f"eda_{split}.{output_format}"
        rows = json.loads(path.read_text()) if output_format == "json" else pq.read_table(path).to_pylist()
        assert len(rows) == 1
        row = rows[0]
        task_id = row["extra_info"]["tools_kwargs"]["task"]["metadata"]["task_id"]
        assert row["prompt"] == preprocess.make_prompt(
            "system instructions", "user instructions", "skill instructions",
        )
        assert f"Repair {task_id}" not in row["prompt"][0]["content"]
        assert "task.md" in row["extra_info"]["tools_kwargs"]["task"]["visible_paths"]


@pytest.mark.parametrize("dataset", [["training_source"] * 5 + ["jpeg_easy_medium_v3"] * 3], indirect=True)
@pytest.mark.parametrize("smoke_test", [False, True])
@pytest.mark.parametrize("output_format", ["json", "parquet"])
def test_smoke_test_selection_matches_data_manifest_and_audit(dataset, tmp_path, smoke_test, output_format):
    if output_format == "parquet":
        pq = pytest.importorskip("pyarrow.parquet")
    root, _ = dataset
    output = tmp_path / "output"
    args = [
        "--dataset-root", str(root), "--output-dir", str(output),
        "--output-format", output_format, "--preview", "0",
    ]
    if smoke_test:
        args.append("--smoke-test")
    assert preprocess.main(args) == 0

    expected_ids = {
        "train": [f"task_{i:04d}" for i in range(1, 5 if smoke_test else 6)],
        "validation": [f"task_{i:04d}" for i in range(6, 8 if smoke_test else 9)],
    }
    with (output / "eda_split_manifest.tsv").open(newline="") as file:
        manifest = list(csv.DictReader(file, delimiter="\t"))
    for split, task_ids in expected_ids.items():
        path = output / f"eda_{split}.{output_format}"
        rows = json.loads(path.read_text()) if output_format == "json" else pq.read_table(path).to_pylist()
        assert [row["extra_info"]["tools_kwargs"]["task"]["metadata"]["task_id"] for row in rows] == task_ids
        assert [row["task_id"] for row in manifest if row["split"] == split] == task_ids

    audit = json.loads((output / "eda_audit.json").read_text())
    assert audit["selection"]["eligible"] == 8
    assert audit["selection"]["selected"] == (6 if smoke_test else 8)
    assert audit["selection"]["smoke_test"] is smoke_test
    assert audit["split"]["counts"] == {split: len(ids) for split, ids in expected_ids.items()}


@pytest.mark.parametrize(("dataset", "split"), [
    (["training_source"] * 3 + ["jpeg_easy_medium_v3"] * 2, "train"),
    (["training_source"] * 4 + ["jpeg_easy_medium_v3"], "validation"),
], indirect=["dataset"])
def test_smoke_test_rejects_insufficient_samples(dataset, tmp_path, capsys, split):
    root, _ = dataset
    output = tmp_path / "output"
    assert preprocess.main([
        "--dataset-root", str(root), "--output-dir", str(output),
        "--output-format", "json", "--smoke-test",
    ]) == 2
    assert f"{split} samples; found" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("name", ["system", "user", "skill"])
def test_parser_no_longer_accepts_prompt_file_overrides(name):
    """提示文件路径固定，不再提供命令行覆盖参数。"""

    parser = preprocess.build_parser()
    option = f"--{name}-prompt-file"
    assert option not in parser.format_help()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--dataset-root", "/dataset", option, "/custom.txt"])
    assert exc.value.code == 2


@pytest.mark.parametrize("name", ["system", "user", "skill"])
@pytest.mark.parametrize("fault", ["missing", "empty"])
def test_invalid_prompt_files_stop_preprocess(dataset, capsys, name, fault):
    root, prompts = dataset
    if fault == "missing":
        prompts[name].unlink()
    else:
        prompts[name].write_text("\ufeff \n", encoding="utf-8")

    assert preprocess.main(["--dataset-root", str(root), "--check-only"]) == 2
    assert "ERROR:" in capsys.readouterr().err
