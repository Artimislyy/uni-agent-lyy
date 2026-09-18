#!/usr/bin/env python3
"""Run one EDA capacity experiment, sample resources, and summarize existing logs.

No torch/Ray imports. `run` needs PyYAML (already needed by training);
`summarize` uses only the standard library. See capacity_benchmark.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LABEL = "uni-agent-capacity-run"
STEP_COLUMNS = {
    "step": "Step",
    "saved_trajectories": "已保存轨迹",
    "eligible_saved_trajectories": "符合有效条件",
    "eligible_saved_per_hour_proxy": "吞吐估算(条/小时)",
    "timing_s/gen": "Rollout+评测等候(s)",
    "timing_s/old_log_prob": "旧策略概率(s)",
    "timing_s/update_actor": "训练更新(s)",
    "timing_s/update_weights": "权重同步(s)",
    "timing_s/step": "Step(s)",
}
RESOURCE_NAMES = {
    "active_sandboxes": "活动沙箱数",
    "sandbox_total_cpu_cores": "沙箱总 CPU 核数",
    "sandbox_total_memory_gib": "沙箱总内存 GiB",
    "node_memory_pressure_gib": "可见节点内存压力 GiB",
}
ERROR_PATTERNS = {
    "timeout_evidence": r"TimeoutError|Request timed out|VERIFIER_TIMEOUT|exec timed out",
    "oom_evidence": r"out of memory|OutOfMemoryError|OOMKilled",
    "verifier_error_evidence": r"VERIFIER_EXECUTION_ERROR|EVAL_PREPARE_ERROR",
}


def read_text(path):
    return path.read_text(errors="replace") if path.exists() else ""


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} exited {result.returncode}: {result.stderr.strip()[:300]}")
    return result.stdout


def numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def gib(value):
    match = re.fullmatch(rf"\s*({NUMBER})\s*([A-Za-z]+)\s*", value)
    if not match:
        return None
    units = {
        "B": 1,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
    }
    scale = units.get(match[2])
    return float(match[1]) * scale / 1024**3 if scale else None


def gpu_sample():
    output = command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for index, gpu_uuid, used, total, utilization in csv.reader(output.splitlines(), skipinitialspace=True):
        rows.append(dict(index=index, uuid=gpu_uuid, used_mib=numeric(used),
                         total_mib=numeric(total), utilization_pct=numeric(utilization)))
    return rows


def docker_sample(binary, run_id):
    ids = command([binary, "ps", "-q", "--filter", f"label={LABEL}={run_id}"]).split()
    if not ids:
        return []
    output = command([binary, "stats", "--no-stream", "--format", "{{json .}}", *ids])
    rows = []
    for line in output.splitlines():
        row = json.loads(line)
        cpu = numeric(row.get("CPUPerc", "").rstrip("%"))
        rows.append(dict(id=row["ID"], name=row.get("Name"), cpu_cores=cpu / 100 if cpu is not None else None,
                         memory_gib=gib(row.get("MemUsage", "").split("/")[0])))
    if len(rows) != len(ids):
        raise RuntimeError("Docker stats returned an incomplete container sample")
    return rows


def node_sample():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(value.split()[0]) / 1024**2
    return {
        "total_gib": values["MemTotal"],
        "available_gib": values["MemAvailable"],
        "used_pressure_gib": values["MemTotal"] - values["MemAvailable"],
        "visible_logical_cpus": os.cpu_count(),
    }


def collect(path, stop, interval, name, function, lock):
    """Each resource has its own thread; slow Docker calls do not delay GPU sampling."""
    with path.open("a", buffering=1) as output:
        while not stop.is_set():
            started = time.monotonic()
            row = {"time": time.time(), "kind": name}
            try:
                row["data"] = function()
            except Exception as exc:
                row["error"] = str(exc)
            row["duration_s"] = time.monotonic() - started
            with lock:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            stop.wait(max(0, interval - row["duration_s"]))


def stats(values):
    values = sorted(v for v in values if v is not None)
    return {
        "samples": len(values),
        "mean": sum(values) / len(values) if values else None,
        "max": max(values) if values else None,
        "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
    }


def resource_summary(path):
    groups, samples, errors = {}, defaultdict(list), Counter()

    def add_group(key, identity, values):
        group = groups.setdefault(key, {**identity, "values": defaultdict(list)})
        for name, value in values.items():
            group["values"][name].append(value)

    if path.exists():
        with path.open() as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors["invalid_sample"] += 1
                    continue
                kind = row["kind"]
                if "error" in row:
                    errors[row["error"]] += 1
                    continue
                if kind == "gpu":
                    for gpu in row["data"]:
                        add_group(
                            ("gpus", gpu["uuid"]),
                            {"index": gpu["index"]},
                            {"memory_mib": gpu["used_mib"], "utilization_pct": gpu["utilization_pct"]},
                        )
                elif kind == "docker":
                    samples["active_sandboxes"].append(len(row["data"]))
                    for field in ("cpu_cores", "memory_gib"):
                        values = [c[field] for c in row["data"]]
                        total = sum(values) if all(v is not None for v in values) else None
                        samples[f"sandbox_total_{field}"].append(total)
                    for container in row["data"]:
                        add_group(
                            ("containers", container["id"]),
                            {"name": container["name"]},
                            {field: container[field] for field in ("cpu_cores", "memory_gib")},
                        )
                elif kind == "node":
                    samples["node_memory_pressure_gib"].append(row["data"]["used_pressure_gib"])
    result = {name: stats(samples[name]) for name in RESOURCE_NAMES}
    result.update(gpus={}, containers={}, collection_errors=dict(errors))
    for (kind, key), group in groups.items():
        values = group.pop("values")
        result[kind][key] = {**group, **{name: stats(items) for name, items in values.items()}}
    return result


def parse_steps(driver):
    steps = {}
    for line in driver.splitlines():
        match = re.search(r"\bstep:(\d+)\s+-\s", line)
        if not match:
            continue
        metrics = {
            name: float(value)
            for name, value in re.findall(rf"\b((?:timing_s|perf|actor/perf)/[\w/]+):({NUMBER})(?=\s|$)", line)
        }
        if "timing_s/step" in metrics:
            steps[int(match[1])] = metrics
    return steps


def summarize(logdir):
    output = logdir / "capacity"
    output.mkdir(parents=True, exist_ok=True)
    driver = ANSI.sub("", read_text(logdir / "driver.log"))
    timings = parse_steps(driver)
    rows, sessions, warnings = [], [], []
    for directory in sorted(logdir.glob("step_*/session-*")):
        if not directory.is_dir():
            continue
        step = int(directory.parent.name.removeprefix("step_"))
        logs = read_text(directory / "framework.log") + "\n" + read_text(directory / "task.log")
        flags = {name: bool(re.search(pattern, logs, re.I)) for name, pattern in ERROR_PATTERNS.items()}
        elapsed = [float(v) for v in re.findall(r"EDA verifier finished in ([\d.]+)s", logs)]
        sessions.append(dict(session_id=directory.name, step=step, verifier_times_s=elapsed, **flags))
        path = directory / "trajectory.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            for traj in data["trajectories"]:
                rows.append(dict(session_id=directory.name, step=step, finished=traj["finished"],
                                 model_tokens=traj["model_token_count"], reward=traj.get("reward_score"),
                                 total_tokens=traj["prompt_len"] + traj["response_len"],
                                 eligible_saved=traj["finished"] is True and traj["model_token_count"] > 0))
        except (ValueError, KeyError, TypeError) as exc:
            warnings.append(f"无法解析 {path}: {exc}")

    steps = []
    tq_error = "TQ write failed" in driver or "TQ write error" in driver
    for step in sorted(set(timings) | {s["step"] for s in sessions}):
        trajectories = [t for t in rows if t["step"] == step]
        metrics = timings.get(step, {})
        elapsed = metrics.get("timing_s/step")
        eligible = sum(t["eligible_saved"] for t in trajectories)
        rate = eligible * 3600 / elapsed if elapsed and trajectories and not tq_error else None
        steps.append(dict(step=step, reported_step_metrics=bool(metrics), saved_trajectories=len(trajectories),
                          eligible_saved_trajectories=eligible, eligible_saved_per_hour_proxy=rate,
                          total_tokens=stats([t["total_tokens"] for t in trajectories]), metrics=metrics))
    completed = [step for step in steps if step["reported_step_metrics"]]
    total_time = sum(step["metrics"]["timing_s/step"] for step in completed)
    complete_coverage = bool(completed) and all(s["saved_trajectories"] for s in completed) and not tq_error
    run_info = json.loads(read_text(output / "run.json") or "null")
    summary = {
        "run": run_info,
        "log_dir": str(logdir),
        "steps": steps,
        "sessions": sessions,
        "trajectories": rows,
        "total_tokens": stats([row["total_tokens"] for row in rows]),
        "model_tokens": stats([row["model_tokens"] for row in rows]),
        "verifier_times_s": stats([elapsed for session in sessions for elapsed in session["verifier_times_s"]]),
        "reported_training_steps": len(completed),
        "step_time_total_s": total_time if completed else None,
        "eligible_saved_per_hour_proxy": sum(s["eligible_saved_trajectories"] for s in completed) * 3600 / total_time
        if complete_coverage and total_time > 0
        else None,
        "session_error_evidence_counts": {flag: sum(s[flag] for s in sessions) for flag in ERROR_PATTERNS},
        "driver_oom_evidence": bool(re.search(ERROR_PATTERNS["oom_evidence"], driver, re.I)),
        "exact_consumed_training_trajectories": None,
        "exact_tool_failure_count": None,
        "resources": resource_summary(output / "resources.jsonl"),
        "warnings": warnings,
        "notes": [
            "吞吐为 finished=True 且有模型 token 的已保存轨迹估算，不证明实际消费或任务答对。",
            "资源覆盖启动到退出的采样窗口；GPU 包含其他进程，节点 /proc 内存未必是 Docker 宿主机视图。",
            "缺失数据标为未知；错误仅计日志证据。计时、采样和统计的详细口径见 capacity_benchmark.md。",
        ],
    }
    write_report(output, summary)
    print(f"容量报告：{output / 'summary.md'}", flush=True)
    return summary


def markdown_table(headers, rows):
    def display(value):
        if value is None:
            return "未采集/无法确认"
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = [" | ".join(map(display, row)) for row in [headers, ["---"] * len(headers), *rows]]
    return "\n".join(f"| {line} |" for line in lines)


def write_report(output, summary):
    write_json(output / "summary.json", summary)
    steps = [{**step, **step["metrics"]} for step in summary["steps"]]
    with (output / "steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=STEP_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(steps)
    res = summary["resources"]
    peaks = [[label, res[key]["max"]] for key, label in RESOURCE_NAMES.items()]
    for gpu in res["gpus"].values():
        peak = gpu["memory_mib"]["max"]
        peaks.append([f"GPU {gpu['index']} 显存 GiB", peak / 1024 if peak is not None else None])
    lengths = [
        [label, *[summary[key][v] for v in ("mean", "p95", "max")]]
        for key, label in (
            ("total_tokens", "总序列 token"),
            ("model_tokens", "模型生成 token"),
            ("verifier_times_s", "验证命令耗时(s)"),
        )
    ]
    errors = [[name, value] for name, value in summary["session_error_evidence_counts"].items()]
    errors.append(["driver.log OOM 证据", summary["driver_oom_evidence"]])
    report = [
        "# EDA 容量测试报告",
        f"日志：`{summary['log_dir']}`；已报告训练 step：{summary['reported_training_steps']}",
        markdown_table(["指标", "数值"], [["有效轨迹吞吐估算(条/小时)", summary["eligible_saved_per_hour_proxy"]]]),
        markdown_table(list(STEP_COLUMNS.values()), [[step.get(key) for key in STEP_COLUMNS] for step in steps]),
        markdown_table(["资源", "观测峰值"], peaks),
        markdown_table(["轨迹与验证", "均值", "P95", "最大值"], lengths),
        markdown_table(["错误日志证据（非完整故障数）", "数量/标记"], errors),
    ]
    run_info = summary["run"]
    if run_info:
        report.append(
            f"测试参数：batch={run_info['batch_size']}，n={run_info['rollout_n']}，"
            f"并发={run_info['concurrency']}；退出码：{run_info.get('returncode')}。"
        )
    report.extend(summary["notes"] + summary["warnings"])
    if res["collection_errors"]:
        report.append("**存在采集错误，详见 summary.json 的 collection_errors。**")
    (output / "summary.md").write_text("\n\n".join(report) + "\n")


def prepare_config(source, target, run_id, docker_binary=None):
    import yaml

    config = yaml.safe_load(source.read_text())
    if not isinstance(config, list):
        raise ValueError("task config 必须是任务列表")
    binaries = set()
    for task in config:
        sandbox = task["sandbox"]
        if sandbox.get("provider") != "docker":
            raise ValueError("容量采集当前只支持 Docker sandbox")
        kwargs = sandbox.setdefault("sandbox_kwargs", {})
        kwargs.setdefault("run_args", []).append(f"--label={LABEL}={run_id}")
        if docker_binary:
            kwargs["docker_binary"] = docker_binary
        binaries.add(kwargs.get("docker_binary", "docker"))
    if len(binaries) != 1:
        raise ValueError("任务需使用同一个 Docker CLI；可用 --docker-bin 指定")
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    target.chmod(0o600)
    return binaries.pop()


def run(args):
    logdir = args.log_dir.resolve()
    if logdir.exists() and any(logdir.iterdir()):
        raise ValueError("请使用一个新的空日志目录，避免混合不同实验")
    output = logdir / "capacity"
    output.mkdir(parents=True)
    run_id = uuid.uuid4().hex
    source = Path(os.environ.get("TASK_CONFIG", HERE / "task_config_claude_code.yaml")).resolve()
    task_config = output / "task_config.yaml"
    binary = prepare_config(source, task_config, run_id, args.docker_bin)
    env = {
        **os.environ,
        "EXP_NAME": logdir.name,
        "AGENT_LOG_DIR": str(logdir),
        "SUBMISSION_DIR": str(output / "submissions"),
        "CKPTS_DIR": str(output / "checkpoints"),
        "TASK_CONFIG": str(task_config),
        "TRAIN_PROMPT_BSZ": str(args.batch_size),
        "N_RESP_PER_PROMPT": str(args.rollout_n),
        "CONCURRENCY": str(args.concurrency),
        "PPO_MINI_BATCH_SIZE": str(args.mini_batch_size),
        "TOTAL_EPOCHS": str(args.steps),
        "SAVE_FREQ": "-1",
    }
    argv = ["bash", str(HERE / "train_qwen3p5_dense.sh"), f"trainer.total_training_steps={args.steps}"]
    fields = ("batch_size", "rollout_n", "concurrency", "steps", "mini_batch_size")
    metadata = {name: getattr(args, name) for name in fields}
    metadata.update(run_id=run_id, started_at=time.time(), sample_interval_s=args.interval,
                    task_config_source=str(source), docker_binary=binary, command=argv,
                    parallel_env_overrides={k: env.get(k) for k in ("TP", "PP", "CP", "GEN_TP")})
    write_json(output / "run.json", metadata)
    stop = threading.Event()
    lock = threading.Lock()
    probes = {"gpu": gpu_sample, "docker": lambda: docker_sample(binary, run_id), "node": node_sample}
    samplers = [threading.Thread(target=collect, args=(output / "resources.jsonl", stop, args.interval, name, fn, lock))
                for name, fn in probes.items()]
    child = None
    previous = {}

    def forward(signum, _frame):
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        for sampler in samplers:
            sampler.start()
        child = subprocess.Popen(argv, cwd=REPO, env=env, start_new_session=True)
        returncode = child.wait()
    finally:
        stop.set()
        for sampler in samplers:
            if sampler.ident is not None:
                sampler.join()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        metadata.update(ended_at=time.time(), returncode=child.returncode if child else None)
        write_json(output / "run.json", metadata)
        summarize(logdir)
    return returncode if returncode >= 0 else 128 - returncode


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    launch = commands.add_parser("run", help="采集资源并运行完整训练 step")
    launch.add_argument("--log-dir", type=Path, required=True)
    for name, default in (("batch-size", 1), ("rollout-n", 1), ("concurrency", 4),
                          ("mini-batch-size", 1), ("steps", 1), ("interval", 5)):
        launch.add_argument(f"--{name}", type=positive_int, default=default)
    launch.add_argument("--docker-bin", help="同时用于训练沙箱和监控的 Docker CLI 路径")
    existing = commands.add_parser("summarize", help="汇总已有日志，不启动训练")
    existing.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        if args.batch_size % args.mini_batch_size:
            parser.error("batch-size 必须能被 mini-batch-size 整除")
        return run(args)
    if not args.log_dir.is_dir():
        parser.error("日志目录不存在")
    summarize(args.log_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
