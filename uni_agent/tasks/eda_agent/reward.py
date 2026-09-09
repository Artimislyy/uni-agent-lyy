"""运行可信的 Innovus 验证器，并把验证结果转换成奖励。"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import PurePosixPath
from typing import Any

from uni_agent.sandbox import SandboxBackend

logger = logging.getLogger(__name__)

_LOG_TAIL_CHARS = 4000
# result.json 的 failure_type 恰好等于以下值时，直接判定违规。
_POLICY_FAILURE_TYPES = {"FORBIDDEN_COMMAND", "POLICY_VIOLATION"}
_PARTIAL_CHECK_GROUPS = (
    ("timing",),
    ("drv", "fanout", "max_transition_clean", "max_capacitance_clean"),
    ("physical", "physical_clean", "placement", "unplaced", "drc", "connectivity"),
)
_OBJECTIVE_CHECKS = {name for group in _PARTIAL_CHECK_GROUPS for name in group}
# 禁令可能被 SUBMISSION_ERROR 包装；只匹配开头的完整标记，避免普通报错中的文字误触发。
# 有些 verifier 不填 failure_type，而是把违规原因写在 reason 字符串里。
_POLICY_REASON = re.compile(
    r"^(?:SUBMISSION_ERROR:\s*)?"
    r"(?:FORBIDDEN_COMMAND|FORBIDDEN_RUNTIME_COMMAND|FORBIDDEN_OPTDESIGN|POLICY_VIOLATION)(?=[:\s]|$)"
)


def is_policy_violation(failure_type: str, reason: str) -> bool:
    """识别可信 verifier 明确报告的违规，普通 Tcl 执行错误不算违规。"""

    return (
        failure_type.strip().upper() in _POLICY_FAILURE_TYPES
        or _POLICY_REASON.match(reason.strip().upper()) is not None
    )


def failure_report(status: str, *, score: float = 0.0, **details: Any) -> dict[str, Any]:
    """生成格式统一的失败报告，方便记录日志和离线分析。"""

    return {
        "resolved": False,
        "status": status,
        "score": score,
        "eval_completed": False,
        "eval_exit_code": None,
        "eval_execution_time": None,
        "eval_stdout_tail": "",
        "eval_stderr_tail": "",
        "verifier_result": None,
        **details,
    }


def score_verifier_result(
    result: dict[str, Any],
    *,
    policy_violation_reward: float = -1.0,
    max_partial_credit: float = 0.5,
) -> tuple[bool, float]:
    """根据 verifier 的结果判断任务是否通过，并计算奖励。

    PASS 得 1 分，明确违规得负分，正常验收失败最多得到 max_partial_credit。
    """

    status = str(result.get("status", "")).upper()
    reason = str(result.get("reason") or result.get("message") or "")
    failure_type = str(result.get("failure_type", ""))
    if status == "PASS":
        return True, 1.0
    # 是否禁用由每条任务的 verifier 决定，这里只映射明确的违规标记。
    if is_policy_violation(failure_type, reason):
        return False, policy_violation_reward
    return False, _partial_credit(result, status, reason, max_partial_credit)


def _partial_credit(result: dict[str, Any], status: str, reason: str, max_score: float) -> float:
    """只对正常验收失败计算核心检查项的部分分。"""

    marker = reason.strip().upper() or status
    normal_failure = marker in {"FAIL", "FINAL_GATE"} or marker.startswith("FAILED_GATES:") #标记是 FAIL、FINAL_GATE，或者以 FAILED_GATES: 开头，就把它认定为正常验收失败。
    normal_failure |= marker.endswith("_GATE") and marker != "INITIAL_GATE"
    checks = result.get("checks")
    #normal_failure = True：确实失败了，而且属于可计算部分分的失败；
    #normal_failure = False：当前状态不属于这种失败，可能是成功、程序异常、数据无效或其他状态。
    #也就是含有FAIL，FINAL_GATE，FAILED_GATES直接算失败，不能计算部分得分，如果checks不是字典类型，也不能计算部分得分。
    if not normal_failure or not isinstance(checks, dict):
        return 0.0

    # 排除非布尔值的检查项、目标检查项和 initial_ 开头的检查项，剩余的检查项只要有False,整个结果就是0.0
    # initial_ 开头的检查项通常是在确认“题目的初始状态是否符合预期”，不是判断“Agent 修得好不好”
    safety_checks = [
        value
        for name, value in checks.items()
        if isinstance(value, bool) and name not in _OBJECTIVE_CHECKS and not name.startswith("initial_")
    ]
    if not all(safety_checks):
        return 0.0

    passed_groups = []
    for group in _PARTIAL_CHECK_GROUPS:
        values = [checks[name] for name in group if isinstance(checks.get(name), bool)]
        if values: #[True, True, False] 
            passed_groups.append(all(values))
    return max_score * sum(passed_groups) / len(passed_groups) if passed_groups else 0.0


async def compute_reward(
    sandbox: SandboxBackend, # 验证沙箱
    *,
    workdir: str, # 沙箱工作目录
    verifier_path: str, # verifier/verify.tcl路径
    answer_path: str, # 模型生成的 repair.tcl 路径
    result_path: str, # verifier 生成的 result.json 路径
    innovus_bin: str, # Innovus 可执行文件路径
    eval_timeout: float, # 验证超时时间
    expected_task_id: str | None = None, # 期望的 task_id，用于检查 result.json 是否属于当前任务
    policy_violation_reward: float = -1.0, # 违规的奖励分数
    max_partial_credit: float = 0.5, # 正常验收失败的最大部分分数
) -> dict[str, Any]:
    """在沙箱中运行 Innovus 验证脚本，并把 result.json 转成奖励报告。"""

    verifier = _remote_path(workdir, verifier_path)  # 验证脚本
    answer = _remote_path(workdir, answer_path)  # 模型生成的 repair.tcl
    result = _remote_path(workdir, result_path)  # verifier 生成的 result.json
    output_dir = str(PurePosixPath(result).parent)  # result.json 所在目录

    prepared = await sandbox.exec(["mkdir", "-p", output_dir])
    if prepared.exit_code != 0:
        return failure_report(
            "EVAL_PREPARE_ERROR",
            eval_exit_code=prepared.exit_code,
            eval_stderr_tail=(prepared.stderr or "")[-_LOG_TAIL_CHARS:],
        )

    # 不用 shell 拼命令，避免路径或可执行文件名被二次解释。
    command = [innovus_bin, "-64", "-no_gui", "-files", verifier] #启动 64 位、无图形界面的 Innovus，执行指定验证脚本。
    env = {
        "TASK_ROOT": workdir,
        "SUBMISSION_TCL": answer,
        "OUTPUT_DIR": output_dir,
        "RESULT_JSON": result,
    }
    started_at = time.perf_counter()
    execution = await sandbox.exec(command, workdir=workdir, env=env, timeout=eval_timeout)
    elapsed = time.perf_counter() - started_at
    # 读取 verifier 的产出，并检查 task_id 是否属于当前任务。
    parsed, read_status = await _read_result(sandbox, result, expected_task_id)
    base = {
        "eval_completed": parsed is not None and execution.exit_code != -1,
        "eval_exit_code": execution.exit_code,
        "eval_execution_time": elapsed,
        "eval_stdout_tail": (execution.stdout or "")[-_LOG_TAIL_CHARS:],
        "eval_stderr_tail": (execution.stderr or "")[-_LOG_TAIL_CHARS:],
        "verifier_result": parsed,
    }
    # exit_code=-1 表示验证超时。
    if execution.exit_code == -1:
        return failure_report("VERIFIER_TIMEOUT", **base)
    # 没有合法 JSON，且退出码非零
    if parsed is None:
        status = "VERIFIER_EXECUTION_ERROR" if execution.exit_code != 0 else read_status
        return failure_report(status, **base)

    resolved, score = score_verifier_result(
        parsed,
        policy_violation_reward=policy_violation_reward,
        max_partial_credit=max_partial_credit,
    )
    # PASS 同时要求进程正常结束；FAIL verifier 有时会有意返回非零码。
    if resolved and execution.exit_code != 0:
        return failure_report("VERIFIER_EXECUTION_ERROR", **base)

    status = str(parsed["status"]).upper()
    logger.info("EDA verifier finished in %.1fs: status=%s score=%.3f", elapsed, status, score)
    return {"resolved": resolved, "status": status, "score": score, **base}


async def _read_result(
    sandbox: SandboxBackend,
    path: str,
    expected_task_id: str | None,
) -> tuple[dict[str, Any] | None, str]:
    """读取并检查 result.json，返回解析结果和读取状态。"""

    try:
        raw = await sandbox.read_file(path)
    except Exception:
        return None, "MISSING_RESULT" #读取抛出异常
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError): #编码错误或 JSON 语法错误返回
        return None, "INVALID_RESULT"
    if not isinstance(value, dict) or not isinstance(value.get("status"), str): #检查 value 是否为字典，并且其中的 status 是否为字符串。只要有一项不符合，就进入 if。
        return None, "INVALID_RESULT"
    if expected_task_id and value.get("task_id") not in {None, expected_task_id}: #如果指定了预期任务 ID，就检查实际任务 ID 是否符合要求。
        return None, "RESULT_TASK_MISMATCH"
    return value, "OK"


def _remote_path(workdir: str, path: str) -> str:
    """把相对路径拼接成沙箱工作目录中的完整路径。"""

    return str(PurePosixPath(workdir) / path)
