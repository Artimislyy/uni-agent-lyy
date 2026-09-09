from __future__ import annotations

import json

import pytest

from uni_agent.tasks.eda_agent import preprocess

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def dataset(tmp_path):
    """构造两道小题，覆盖真实读取、划分和输出流程，无需 Innovus。"""

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
    for number, source in enumerate(["training_source", "jpeg_easy_medium_v3"], start=1):
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
    prompt = preprocess.make_prompt(" system ", " user ", " task ", skill)

    expected_content = "\n\n".join([
        "## EDA agent operating instructions\n\nsystem",
        f"## Innovus ECO closure skill (SKILL.md)\n\n{skill}",
        "## Task request\n\nuser",
        "## Current task specification (task.md)\n\ntask",
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
            "system instructions", "user instructions", f"Repair {task_id}", "skill instructions",
        )


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
