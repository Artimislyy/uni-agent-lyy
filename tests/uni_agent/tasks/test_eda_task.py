from __future__ import annotations

import asyncio
import json

import pytest

from uni_agent.agents import AgentResult
from uni_agent.sandbox import ExecResult
from uni_agent.tasks.eda_agent.reward import compute_reward, score_verifier_result
from uni_agent.tasks.eda_agent.task import EDATask, EDATaskConfig, _infer_status


class _FakeSandbox:
    """Small in-memory sandbox used to verify lifecycle and visibility boundaries."""

    def __init__(self, name: str, events: list[str], *, task_id: str = "task_0001", innovus_exit: int = 0):
        self.name = name
        self.events = events
        self.task_id = task_id
        self.innovus_exit = innovus_exit
        self.files: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.commands: list[tuple[list[str], dict | None]] = []

    async def __aenter__(self):
        self.events.append(f"start:{self.name}")
        return self

    async def __aexit__(self, *_):
        self.events.append(f"stop:{self.name}")

    async def exec(self, argv, *, timeout=None, workdir=None, env=None):
        argv = list(argv)
        self.commands.append((argv, env))
        if argv[:2] == ["test", "-f"]:
            code = 0 if argv[2] in self.files else 1
            return ExecResult(code, "", "")
        if argv[:3] == ["test", "!", "-L"]:
            return ExecResult(0, "", "")
        if argv[:3] == ["stat", "-c", "%s"]:
            data = self.files.get(argv[3])
            return ExecResult(0 if data is not None else 1, str(len(data)) if data is not None else "", "")
        if argv and argv[0] == "innovus":
            assert env is not None
            self.files[env["RESULT_JSON"]] = json.dumps(
                {"task_id": self.task_id, "status": "PASS", "reason": "PASS"}
            ).encode()
            return ExecResult(self.innovus_exit, "verifier done", "")
        return ExecResult(0, "", "")

    async def upload(self, _local_path, remote_path):
        self.uploads.append(str(remote_path))

    async def write_file(self, path, content):
        self.files[str(path)] = content.encode() if isinstance(content, str) else content

    async def read_file(self, path):
        if str(path) not in self.files:
            raise FileNotFoundError(path)
        return self.files[str(path)]


class _FakeAgent:
    def __init__(self, finished: bool = True, submission: bytes | None = b"# model answer\n"):
        self.finished = finished
        self.submission = submission

    async def run(self, *, sandbox, messages, workdir):
        assert messages == [{"role": "user", "content": "repair it"}]
        assert not any("/verifier" in path or "/reference" in path for path in sandbox.uploads)
        if self.submission is not None:
            await sandbox.write_file(f"{workdir}/repair.tcl", self.submission)
        return AgentResult(finished=self.finished, info={"exit_code": 0 if self.finished else -1})


def _make_task_tree(tmp_path):
    dataset = tmp_path / "dataset"
    task = dataset / "tasks/design/task_0001"
    (task / "initial_state/design.enc.dat").mkdir(parents=True)
    (task / "initial_state/reports").mkdir()
    (task / "verifier").mkdir()
    (task / "reference").mkdir()
    (task / "task.md").write_text("task")
    (task / "initial_state/design.enc").write_text("checkpoint")
    (task / "initial_state/design.enc.dat/db.bin").write_bytes(b"db")
    (task / "verifier/verify.tcl").write_text("# verifier")
    (task / "verifier/common.tcl").write_text("# helper")
    (task / "reference/repair.tcl").write_text("# golden")
    return dataset


def _config() -> EDATaskConfig:
    return EDATaskConfig(
        task_root="tasks/design/task_0001",
        sandbox={"provider": "docker", "image": "fake"},
        sandbox_image_env=None,
        submission_dir=None,
        submission_dir_env=None,
        agent={
            "name": "claude_code",
            "model": {"base_url": "http://policy/v1", "model_name": "policy"},
        },
        prompt=[{"role": "user", "content": "repair it"}],
        visible_paths=["task.md", "initial_state/design.enc", "initial_state/design.enc.dat"],
        hidden_paths=["verifier", "reference", "initial_state/reports"],
        metadata={"task_id": "task_0001"},
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("agent_finished", [True, False])
def test_task_uses_two_sandboxes_and_only_transfers_submission(monkeypatch, tmp_path, agent_finished):
    dataset = _make_task_tree(tmp_path)
    monkeypatch.setenv("EDA_DATASET_ROOT", str(dataset))
    events: list[str] = []
    agent_sandbox = _FakeSandbox("agent", events)
    eval_sandbox = _FakeSandbox("eval", events)
    sandboxes = iter([agent_sandbox, eval_sandbox])
    task = EDATask(_config())
    monkeypatch.setattr(task, "build_sandbox", lambda: next(sandboxes))
    monkeypatch.setattr(task, "build_agent", lambda: _FakeAgent(agent_finished))

    result = asyncio.run(task.run())

    assert result.reward == 1.0
    assert result.accuracy == 1.0
    assert result.finished is agent_finished
    assert events == ["start:agent", "stop:agent", "start:eval", "stop:eval"]
    assert not any("/verifier" in path or "/reference" in path for path in agent_sandbox.uploads)
    assert any("/verifier" in path for path in eval_sandbox.uploads)
    assert not any("/reference" in path or "/initial_state/reports" in path for path in eval_sandbox.uploads)
    assert b"# model answer" in next(data for path, data in eval_sandbox.files.items() if path.endswith("repair.tcl"))
    assert result.extra_info["isolation"]["strategy"] == "fresh_sandbox"
    assert result.extra_info["infer_status"] == ("FINISHED" if agent_finished else "INFER_TIMEOUT")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "finished, info, expected",
    [
        (True, {"exit_code": 0}, "FINISHED"),
        (None, {}, "FINISHED"),
        (False, {"exit_code": -1}, "INFER_TIMEOUT"),
        (False, {"termination_reason": "request timeout"}, "INFER_TIMEOUT"),
        (False, {"exit_code": 1}, "INFER_INCOMPLETE"),
    ],
)
def test_infer_status(finished, info, expected):
    assert _infer_status(finished, info) == expected


@pytest.mark.cpu
@pytest.mark.level0
def test_task_persists_submission_on_host(monkeypatch, tmp_path):
    dataset = _make_task_tree(tmp_path)
    submission_root = tmp_path / "submissions"
    monkeypatch.setenv("EDA_DATASET_ROOT", str(dataset))

    events: list[str] = []
    sandboxes = iter([_FakeSandbox("agent", events), _FakeSandbox("eval", events)])
    config = _config()
    config.submission_dir = str(submission_root)
    task = EDATask(config)
    monkeypatch.setattr(task, "build_sandbox", lambda: next(sandboxes))
    monkeypatch.setattr(task, "build_agent", lambda: _FakeAgent())

    result = asyncio.run(task.run())

    saved_path = submission_root / "task_0001"
    saved_files = list(saved_path.glob("*/repair.tcl"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == b"# model answer\n"
    assert result.extra_info["submission_path"] == str(saved_files[0])


@pytest.mark.cpu
@pytest.mark.level0
def test_submission_dir_env_overrides_yaml_default(monkeypatch, tmp_path):
    dataset = _make_task_tree(tmp_path)
    yaml_root = tmp_path / "yaml-submissions"
    env_root = tmp_path / "env-submissions"
    monkeypatch.setenv("EDA_DATASET_ROOT", str(dataset))
    monkeypatch.setenv("EDA_SUBMISSION_DIR", str(env_root))

    events: list[str] = []
    sandboxes = iter([_FakeSandbox("agent", events), _FakeSandbox("eval", events)])
    config = _config()
    config.submission_dir = str(yaml_root)
    config.submission_dir_env = "EDA_SUBMISSION_DIR"
    task = EDATask(config)
    monkeypatch.setattr(task, "build_sandbox", lambda: next(sandboxes))
    monkeypatch.setattr(task, "build_agent", lambda: _FakeAgent())

    result = asyncio.run(task.run())

    saved_files = list((env_root / "task_0001").glob("*/repair.tcl"))
    assert len(saved_files) == 1
    assert result.extra_info["submission_path"] == str(saved_files[0])
    assert not yaml_root.exists()


@pytest.mark.cpu
@pytest.mark.level0
def test_config_rejects_path_escape_and_local_provider():
    values = _config().model_dump()
    values["task_root"] = "../secret"
    with pytest.raises(ValueError, match="safe relative path"):
        EDATaskConfig(**values)

    values = _config().model_dump()
    values["sandbox"] = {"provider": "local"}
    with pytest.raises(ValueError, match="isolated sandbox"):
        EDATaskConfig(**values)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("policy_reward", [-1.0, -0.25, 0.0])
@pytest.mark.parametrize(
    "details, is_violation",
    [
        ({"failure_type": "FORBIDDEN_COMMAND"}, True),
        ({"failure_type": "POLICY_VIOLATION"}, True),
        ({"reason": "POLICY_VIOLATION:checkpoint modified"}, True),
        ({"reason": "FORBIDDEN_COMMAND"}, True),
        ({"reason": "FORBIDDEN_COMMAND:optDesign"}, True),
        ({"reason": "FORBIDDEN_COMMAND:restoreDesign"}, True),
        ({"reason": "SUBMISSION_ERROR:FORBIDDEN_COMMAND:exit"}, True),
        ({"reason": "SUBMISSION_ERROR:FORBIDDEN_RUNTIME_COMMAND:exit"}, True),
        ({"reason": "SUBMISSION_ERROR:FORBIDDEN_OPTDESIGN targeted-only task"}, True),
        ({"message": " submission_error: forbidden_runtime_command:exit "}, True),
        ({"failure_type": "SUBMISSION_ERROR", "reason": 'invalid command name "foo"'}, False),
        ({"reason": "SUBMISSION_ERROR:wrong # args"}, False),
        ({"reason": "SUBMISSION_ERROR"}, False),
        ({"reason": 'SUBMISSION_ERROR:invalid command name "FORBIDDEN_COMMAND_helper"'}, False),
        ({"reason": "FORBIDDEN_COMMAND_helper failed"}, False),
        ({"reason": "FINAL_GATE"}, False),
        ({"reason": "INITIAL_GATE"}, False),
        ({"reason": "TIMING_OR_DRV_GATE"}, False),
        ({"reason": "PHYSICAL_GATE"}, False),
        ({"reason": "INITIALIZATION_ERROR"}, False),
        ({"reason": "VERIFIER_ERROR"}, False),
        ({"reason": "MISSING_SUBMISSION"}, False),
    ],
)
def test_reward_only_penalizes_explicit_policy_violation(details, is_violation, policy_reward):
    """真实 verifier 的禁令包装格式会扣分，普通错误和文字提及不会。"""

    result = {"status": "FAIL", **details}
    expected_score = policy_reward if is_violation else 0.0
    assert score_verifier_result(result, policy_violation_reward=policy_reward) == (False, expected_score)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "result, expected",
    [
        (
            {
                "status": "FAIL",
                "reason": "FINAL_GATE",
                "checks": {"initial_failure": True, "timing": True, "drv": False, "physical": True},
            },
            0.5 * 2 / 3,
        ),
        (
            {"status": "FAIL", "checks": {"timing": True, "drv": False, "physical": True}},
            0.5 * 2 / 3,
        ),
        (
            {"status": "FINAL_GATE", "checks": {"timing": True, "drv": False, "physical": True}},
            0.5 * 2 / 3,
        ),
        (
            {
                "status": "FAIL",
                "reason": "FINAL_GATE",
                "checks": {"timing": True, "drv": True, "physical": True, "functional": False},
            },
            0.0,
        ),
        (
            {
                "status": "FAIL",
                "reason": "INITIAL_GATE",
                "checks": {"timing": True, "drv": True, "physical": True},
            },
            0.0,
        ),
        ({"status": "FAIL", "reason": "FINAL_GATE", "checks": {"unknown": True}}, 0.0),
    ],
)
def test_partial_credit_uses_only_safe_objective_checks(result, expected):
    resolved, score = score_verifier_result(result)

    assert resolved is False
    assert score == pytest.approx(expected)


@pytest.mark.cpu
@pytest.mark.level0
def test_policy_violation_never_receives_partial_credit():
    result = {
        "status": "FAIL",
        "reason": "FORBIDDEN_COMMAND:exit",
        "checks": {"timing": True, "drv": True, "physical": True},
    }

    assert score_verifier_result(result, policy_violation_reward=-0.25) == (False, -0.25)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("agent_finished", [True, False])
@pytest.mark.parametrize(
    "fault, submission, expected_status",
    [
        ("missing", None, "MISSING_SUBMISSION"),
        ("empty", b"", "EMPTY_SUBMISSION"),
        ("too_large", b"x" * 17, "SUBMISSION_TOO_LARGE"),
        ("symlink", b"# answer", "UNSAFE_SUBMISSION"),
        ("symlink", None, "UNSAFE_SUBMISSION"),
        ("stat_error", b"# answer", "UNREADABLE_SUBMISSION"),
        ("invalid_stat", b"# answer", "UNREADABLE_SUBMISSION"),
        ("read_error", b"# answer", "UNREADABLE_SUBMISSION"),
        ("size_changed", b"# answer", "SUBMISSION_SIZE_CHANGED"),
        ("symlink_check_timeout", b"# answer", "UNREADABLE_SUBMISSION"),
        ("file_check_timeout", b"# answer", "UNREADABLE_SUBMISSION"),
    ],
)
def test_submission_failure_rewards_and_lifecycle(
    monkeypatch, tmp_path, agent_finished, fault, submission, expected_status
):
    """覆盖六种读取失败；超时不放大惩罚，失败后关闭容器并跳过验证。"""

    monkeypatch.setenv("EDA_DATASET_ROOT", str(_make_task_tree(tmp_path)))
    events: list[str] = []
    sandbox = _FakeSandbox("agent", events)
    original_exec = sandbox.exec
    original_read = sandbox.read_file

    async def exec_with_fault(argv, **kwargs):
        if argv[:3] == ["test", "!", "-L"]:
            if fault == "symlink":
                return ExecResult(1, "", "")
            if fault == "symlink_check_timeout":
                return ExecResult(-1, "", "timeout")
        if argv[:2] == ["test", "-f"] and fault == "file_check_timeout":
            return ExecResult(-1, "", "timeout")
        if argv[:3] == ["stat", "-c", "%s"]:
            if fault == "stat_error":
                return ExecResult(1, "", "cannot stat")
            if fault == "invalid_stat":
                return ExecResult(0, "invalid size", "")
        return await original_exec(argv, **kwargs)

    async def read_with_fault(path):
        if fault == "read_error":
            raise OSError("cannot read submission")
        content = await original_read(path)
        return content + b"extra" if fault == "size_changed" else content

    monkeypatch.setattr(sandbox, "exec", exec_with_fault)
    monkeypatch.setattr(sandbox, "read_file", read_with_fault)
    config = _config()
    config.max_submission_bytes = 16
    config.policy_violation_reward = -0.25
    task = EDATask(config)
    sandboxes = iter([sandbox])
    monkeypatch.setattr(task, "build_sandbox", lambda: next(sandboxes))
    monkeypatch.setattr(task, "build_agent", lambda: _FakeAgent(agent_finished, submission))

    result = asyncio.run(task.run())

    assert result.reward == (-0.25 if expected_status == "UNSAFE_SUBMISSION" else 0.0)
    assert result.accuracy == 0.0
    assert result.finished is agent_finished
    assert result.extra_info["score"] == result.reward
    assert result.extra_info["status"] == expected_status
    assert result.extra_info["submission_status"] == expected_status
    assert result.extra_info["infer_status"] == ("FINISHED" if agent_finished else "INFER_TIMEOUT")
    assert result.extra_info["agent_info"]["exit_code"] == (0 if agent_finished else -1)
    assert result.extra_info["eval_completed"] is False
    assert events == ["start:agent", "stop:agent"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("exit_code, status", [(2, "VERIFIER_EXECUTION_ERROR"), (-1, "VERIFIER_TIMEOUT")])
def test_nonzero_exit_cannot_turn_pass_result_into_reward(exit_code, status):
    sandbox = _FakeSandbox("eval", [], innovus_exit=exit_code)

    async def run():
        await sandbox.write_file("/work/repair.tcl", b"# answer")
        return await compute_reward(
            sandbox,
            workdir="/work",
            verifier_path="verifier/verify.tcl",
            answer_path="repair.tcl",
            result_path="verifier_output/result.json",
            innovus_bin="innovus",
            eval_timeout=10,
            expected_task_id="task_0001",
        )

    report = asyncio.run(run())

    assert report["resolved"] is False
    assert report["score"] == 0.0
    assert report["status"] == status
