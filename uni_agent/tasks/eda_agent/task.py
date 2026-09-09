"""使用独立 Agent 沙箱和验证沙箱执行 EDA 任务。"""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path, PurePosixPath

from pydantic import Field, field_validator, model_validator

from uni_agent.sandbox import SandboxBackend

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import compute_reward, failure_report

logger = logging.getLogger(__name__)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TIMEOUT_FIELDS = ("exit_status", "termination_reason", "reason", "stderr_tail", "stdout_tail", "error")


class EDATaskConfig(TaskConfig):
    """保存 EDA 样本路径和运行时配置。"""

    name: str = "eda"
    task_root: str = Field(description="Task path relative to dataset_root_env.")
    dataset_root_env: str = "EDA_DATASET_ROOT"
    visible_paths: list[str] = Field(default_factory=list)
    hidden_paths: list[str] = Field(default_factory=list)
    verifier_path: str = "verifier/verify.tcl"
    answer_path: str = "repair.tcl"
    result_path: str = "verifier_output/result.json"

    remote_workspace: str = "/workspace/uni-agent-eda"
    innovus_bin: str = "innovus"
    eval_timeout: float = Field(default=7200.0, gt=0)
    max_submission_bytes: int = Field(default=1024 * 1024, gt=0) #提交文件大小上限，默认 1 MiB
    policy_violation_reward: float = Field(default=-1.0, ge=-1.0, le=0.0) #策略违规奖励
    max_partial_credit: float = Field(default=0.5, ge=0.0, lt=1.0) #最大部分奖励，默认 0.5
    sandbox_image_env: str | None = "EDA_SANDBOX_IMAGE" #沙箱镜像环境变量
    submission_dir: str | None = None #宿主机 Tcl 留档目录
    submission_dir_env: str | None = "EDA_SUBMISSION_DIR" #宿主机 Tcl 留档目录环境变量

    @field_validator("task_root", "verifier_path", "answer_path", "result_path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        """检查单个配置路径是否为安全的相对路径。"""

        return _relative_path(value)

    @field_validator("visible_paths", "hidden_paths")
    @classmethod
    def _validate_path_list(cls, values: list[str]) -> list[str]:
        """检查路径列表是否合法且没有重复项。"""

        normalized = [_relative_path(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("path lists must not contain duplicates")
        return normalized

    @field_validator("remote_workspace")
    @classmethod
    def _validate_remote_workspace(cls, value: str) -> str:
        """确保沙箱工作目录是根目录以下的安全绝对路径。"""

        path = PurePosixPath(value)
        if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
            raise ValueError("remote_workspace must be a safe absolute path below /")
        return path.as_posix().rstrip("/")

    @model_validator(mode="after")
    def _validate_visibility(self) -> EDATaskConfig:
        """检查文件可见范围、沙箱类型和环境变量配置。"""

        if not self.visible_paths:
            raise ValueError("visible_paths must not be empty")
        if not self.hidden_paths:
            raise ValueError("hidden_paths must not be empty")
        if not any(_contains(parent, self.verifier_path) for parent in self.hidden_paths):
            raise ValueError("verifier_path must be covered by hidden_paths")
        if any(_overlap(visible, hidden) for visible in self.visible_paths for hidden in self.hidden_paths):
            raise ValueError("visible_paths and hidden_paths must not overlap")
        generated_paths = (self.answer_path, self.result_path)
        protected_paths = [*self.visible_paths, *self.hidden_paths]
        if any(_overlap(generated, protected) for generated in generated_paths for protected in protected_paths):
            raise ValueError("answer_path and result_path must not overlap input assets")
        if _overlap(self.answer_path, self.result_path):
            raise ValueError("answer_path and result_path must not overlap")
        if self.sandbox.provider == "local":
            raise ValueError("EDA tasks require an isolated sandbox; provider='local' is unsafe")
        if not _ENV_NAME.fullmatch(self.dataset_root_env):
            raise ValueError("dataset_root_env is not a valid environment variable name")
        if self.sandbox_image_env and not _ENV_NAME.fullmatch(self.sandbox_image_env):
            raise ValueError("sandbox_image_env is not a valid environment variable name")
        if self.submission_dir_env and not _ENV_NAME.fullmatch(self.submission_dir_env):
            raise ValueError("submission_dir_env is not a valid environment variable name")

        # 专有 Innovus 镜像通常由部署环境决定，允许训练脚本通过环境变量覆盖 YAML 占位值。
        if self.sandbox_image_env and (image := os.getenv(self.sandbox_image_env)):
            self.sandbox.image = image
        return self


@register_task("eda")
class EDATask(Task):
    """在 Agent 沙箱生成 repair.tcl，再到干净沙箱中验证。"""

    name = "eda"
    config_model = EDATaskConfig

    async def run(self) -> TaskResult:
        """依次完成任务准备、Agent 执行、答案提取和独立验证。"""

        cfg: EDATaskConfig = self.config  # type: ignore[assignment]
        task_root = _resolve_task_root(cfg) #检查环境变量有值、数据集和任务目录存
        _check_assets(task_root, [*cfg.visible_paths, *cfg.hidden_paths]) # 检查所需文件都存在，缺了就报错

        # 第一段生命周期：模型只能看到 visible_paths。
        agent_root = _new_workspace(cfg.remote_workspace, "agent") #生成带 UUID 的路径字符串
        async with self.build_sandbox() as agent_sandbox: #创建一次性容器
            await _stage(agent_sandbox, task_root, cfg.visible_paths, agent_root) #创建目录，上传dat数据集
            agent_result = await self.build_agent().run(
                sandbox=agent_sandbox,
                messages=cfg.prompt,
                workdir=agent_root,
            ) #启动 Agent（Claude Code），在容器里干活
            infer_status = _infer_status(agent_result.finished, agent_result.info)
            common_info = {
                "task_root": str(task_root),
                "task_id": cfg.metadata.get("task_id"),
                "submission_path": None,
                "submission_status": "NOT_ATTEMPTED",
                "infer_completed": agent_result.finished is True,
                "infer_status": infer_status,
                "agent_info": agent_result.info,
                "isolation": {
                    "strategy": "fresh_sandbox",
                    "submission_transfer": "memory_only",
                    "reference_uploaded": False,
                },
            }
            # 只有明确正常结束才读取、留档答案；提前返回也会退出上下文并清理 A。
            if agent_result.finished is not True:
                return TaskResult(
                    reward=0.0,
                    accuracy=0.0,
                    finished=False,
                    extra_info=failure_report(infer_status, **common_info),
                )

            submission, submission_status, submission_path = await _read_submission(
                agent_sandbox,
                _remote_path(agent_root, cfg.answer_path),
                cfg.max_submission_bytes,
                local_dir=cfg.submission_dir,
                local_dir_env=cfg.submission_dir_env,
                task_id=str(cfg.metadata.get("task_id") or PurePosixPath(cfg.task_root).name),
            )#从容器里读出答案文件 repair.tcl，并检查大小不超过上限（1MB）。
        # 离开上下文后，Claude Code、Innovus 进程和脏工作区一起被销毁。

        common_info.update(
            submission_path=str(submission_path) if submission_path else None,
            submission_status=submission_status,
        )
        if submission is None:
            # 只有符号链接等明确不安全提交才扣分；缺失、空文件、超大和读取故障均为 0。
            score = cfg.policy_violation_reward if submission_status == "UNSAFE_SUBMISSION" else 0.0
            report = failure_report(
                submission_status,
                score=score,
                **common_info,
            )
            return TaskResult(
                reward=score,
                accuracy=0.0,
                finished=agent_result.finished,
                extra_info=report,
            )

        common_info["isolation"]["submission_persisted"] = submission_path is not None

        # 第二段生命周期：从宿主数据集重新复制原始 checkpoint，绝不复用模型改过的文件。
        eval_root = _new_workspace(cfg.remote_workspace, "eval")
        verifier_bundle = PurePosixPath(cfg.verifier_path).parent.as_posix()
        eval_paths = [*cfg.visible_paths, verifier_bundle if verifier_bundle != "." else cfg.verifier_path]
        async with self.build_sandbox() as eval_sandbox:
            await _stage(eval_sandbox, task_root, eval_paths, eval_root)
            await eval_sandbox.write_file(_remote_path(eval_root, cfg.answer_path), submission)
            report = await compute_reward(
                eval_sandbox,
                workdir=eval_root,
                verifier_path=cfg.verifier_path,
                answer_path=cfg.answer_path,
                result_path=cfg.result_path,
                innovus_bin=cfg.innovus_bin,
                eval_timeout=cfg.eval_timeout,
                expected_task_id=cfg.metadata.get("task_id"),
                policy_violation_reward=cfg.policy_violation_reward,
                max_partial_credit=cfg.max_partial_credit,
            )

        report.update(common_info)
        return TaskResult(
            reward=float(report["score"]), #训练分数
            accuracy=1.0 if report["resolved"] else 0.0, #是否通过验证
            finished=agent_result.finished, #Agent 是否正常结束
            extra_info=report, #额外信息
        )


async def _stage(sandbox: SandboxBackend, task_root: Path, paths: list[str], remote_root: str) -> None:
    """把宿主机中的指定任务文件上传到沙箱工作目录。"""

    created = await sandbox.exec(["mkdir", "-p", remote_root])
    if created.exit_code != 0:
        raise RuntimeError(f"cannot create sandbox workspace {remote_root}: {created.stderr}")
    for relative in paths:
        source = _asset_path(task_root, relative)
        destination = _remote_path(remote_root, relative)
        parent = str(PurePosixPath(destination).parent)
        prepared = await sandbox.exec(["mkdir", "-p", parent])
        if prepared.exit_code != 0:
            raise RuntimeError(f"cannot create sandbox directory {parent}: {prepared.stderr}")
        await sandbox.upload(source, destination)


async def _read_submission(
    sandbox: SandboxBackend,
    path: str,
    size_limit: int,
    *,
    local_dir: str | None,
    local_dir_env: str | None,
    task_id: str,
) -> tuple[bytes | None, str, Path | None]:
    """安全读取 repair.tcl，并将合法文件直接保存到宿主机。"""

    # 先判符号链接，目标不存在的链接也属于不安全提交。
    regular = await sandbox.exec(["test", "-f", path]) #在沙箱环境中检查 path 是否为普通文件，并等待检查结束
    not_symlink = await sandbox.exec(["test", "!", "-L", path]) #在沙箱中检查 path 不是符号链接，并把执行结果保存到 not_symlink。
    if not_symlink.exit_code == 1:
        return None, "UNSAFE_SUBMISSION", None
    if not_symlink.exit_code != 0 or regular.exit_code not in {0, 1}: #这行代码的作用是：检查前面两个文件检测命令是否出现异常；只要有一项异常，就进入 if
        return None, "UNREADABLE_SUBMISSION", None
    if regular.exit_code == 1:
        return None, "MISSING_SUBMISSION", None #普通文件检查不通过，例如没有生成答案
    #大小校验
    size_result = await sandbox.exec(["stat", "-c", "%s", path]) #在沙箱中获取 path 对应文件的大小，并将执行结果保存到 size_result。
    if size_result.exit_code != 0:
        return None, "UNREADABLE_SUBMISSION", None # stat 读不出大小
    try:
        size = int(size_result.stdout.strip())
    except ValueError:
        return None, "UNREADABLE_SUBMISSION", None #无法读取,（文件瞬间被删了？权限异常？stat 输出畸形？）
    if size == 0:
        return None, "EMPTY_SUBMISSION", None #空文件, 大小 = 0
    if size > size_limit:
        return None, "SUBMISSION_TOO_LARGE", None #文件大小超过限制
    try:
        #读取内容(防 TOCTOU 竞态)，“检查时刻”和“使用时刻”不一致
        content = await sandbox.read_file(path)
    except Exception:
        return None, "UNREADABLE_SUBMISSION", None  # 读取失败，尚不能判断内容大小是否变化。
    if len(content) != size or len(content) > size_limit:
        return None, "SUBMISSION_SIZE_CHANGED", None

    #写入本地
    local_path = None
    # 环境变量有值时覆盖 YAML 中的默认保存目录。
    env_dir = os.getenv(local_dir_env) if local_dir_env else None
    root_value = env_dir or local_dir
    if root_value:
        root = Path(os.path.expandvars(root_value)).expanduser().resolve()
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._") or "unknown-task"
        output_dir = root / safe_task_id / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=False)
        local_path = output_dir / PurePosixPath(path).name
        local_path.write_bytes(content)

    return content, "OK", local_path


def _resolve_task_root(cfg: EDATaskConfig) -> Path:
    """根据环境变量和样本相对路径找到宿主机任务目录。"""

    dataset_value = os.getenv(cfg.dataset_root_env)
    if not dataset_value:
        raise ValueError(f"set {cfg.dataset_root_env} to the dataset_innovus_19_10 directory")
    dataset_root = Path(os.path.expandvars(dataset_value)).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"EDA dataset root does not exist: {dataset_root}")
    task_root = (dataset_root / cfg.task_root).resolve()
    if not task_root.is_relative_to(dataset_root):
        raise ValueError(f"task_root escapes the dataset root: {cfg.task_root!r}")
    if not task_root.is_dir():
        raise FileNotFoundError(f"EDA task root does not exist: {task_root}")
    return task_root


def _check_assets(task_root: Path, paths: list[str]) -> None:
    """检查任务运行所需的文件和目录是否全部存在。"""

    for relative in dict.fromkeys(paths):
        path = _asset_path(task_root, relative)
        if not path.exists():
            raise FileNotFoundError(f"EDA task asset does not exist: {path}")


def _asset_path(task_root: Path, relative: str) -> Path:
    """解析任务资源路径，并禁止路径逃出任务目录。"""

    path = (task_root / relative).resolve()
    if not path.is_relative_to(task_root):
        raise ValueError(f"task asset escapes task_root: {relative!r}")
    return path


def _relative_path(value: str) -> str:
    """校验并规范化安全的相对路径。"""

    path = PurePosixPath(value)
    if not value or path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError(f"expected a safe relative path, got {value!r}")
    return path.as_posix()


def _contains(parent: str, child: str) -> bool:
    """判断 child 是否等于 parent 或位于其目录下。"""

    parent_path, child_path = PurePosixPath(parent), PurePosixPath(child)
    return parent_path == child_path or parent_path in child_path.parents


def _overlap(left: str, right: str) -> bool:
    """判断两个路径是否相同或存在父子包含关系。"""

    return _contains(left, right) or _contains(right, left)


def _new_workspace(base: str, phase: str) -> str:
    """生成带 UUID 的沙箱工作目录，避免不同任务相互污染。"""

    return f"{base}/{phase}-{uuid.uuid4().hex}"


def _infer_status(finished: bool | None, info: object) -> str:
    """区分 Agent 正常结束、超时和其他未完成情况。"""

    # 只有 True 才确认正常结束；None 表示未确认完成。
    if finished is True:
        return "FINISHED"
    if isinstance(info, dict) and (
        info.get("exit_code") == -1
        or any("timeout" in str(info.get(field, "")).lower() for field in _TIMEOUT_FIELDS)
    ):
        return "INFER_TIMEOUT"
    return "INFER_INCOMPLETE"


def _remote_path(workdir: str, path: str) -> str:
    """把相对路径拼接成沙箱内的完整路径。"""

    return str(PurePosixPath(workdir) / path)
