#!/usr/bin/env python3
"""Utilities for per-run experiment tracking."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_INDEX_FIELDS = [
    "run_name",
    "model_alias",
    "method",
    "config_path",
    "output_dir",
    "launch_time",
    "git_commit",
    "launcher",
    "tag",
    "status",
    "temp_config_path",
    "deepspeed_config",
]


def _strip_inline_comment(value: str) -> str:
    if "#" in value:
        value = value.split("#", 1)[0]
    return value.strip().strip('"').strip("'")


def _read_yaml_scalar(config_path: str, key: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$")
    text = Path(config_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            return _strip_inline_comment(match.group(1))
    return None


def infer_model_alias(config_path: str) -> str:
    path = Path(config_path)
    stem = path.stem.lower()
    filename_match = re.search(r"(qwen[0-9.]*[_-][0-9]+[bm])", stem)
    if filename_match:
        return filename_match.group(1).replace("_", "-")

    model_path = _read_yaml_scalar(config_path, "model_name_or_path")
    if model_path:
        base = Path(model_path).name.lower().replace("_", "-")
        name_match = re.search(r"(qwen[0-9.]*-[0-9]+[bm])", base)
        if name_match:
            return name_match.group(1)

    return "unknown-model"


def infer_method(config_path: str) -> str:
    stem = Path(config_path).stem.lower()
    if "qlora" in stem:
        return "qlora"

    finetuning_type = (_read_yaml_scalar(config_path, "finetuning_type") or "").lower()
    quantization_bit = _read_yaml_scalar(config_path, "quantization_bit")

    if finetuning_type == "lora" and quantization_bit:
        return "qlora"
    if finetuning_type in {"lora", "qlora"}:
        return finetuning_type
    return finetuning_type or "unknown"


def make_run_name(model_alias: str, method: str, tag: str | None = None) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    safe_tag = (tag or "run").strip().replace(" ", "_")
    safe_tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", safe_tag)
    return f"{timestamp}_{safe_tag}"


def build_output_dir(base: str = "runs", model_alias: str = "unknown-model", method: str = "unknown", run_name: str = "run") -> str:
    return str(Path(base) / model_alias / method / run_name)


def snapshot_config(config_path: str, output_dir: str) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "config_used.yaml"
    shutil.copy2(config_path, target)
    return str(target)


def _run_git_command(args: list[str]) -> str:
    try:
        output = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def write_git_info(output_dir: str) -> str:
    commit = _run_git_command(["git", "rev-parse", "HEAD"])
    branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git_command(["git", "status", "--porcelain"])
    dirty = "true" if status not in {"", "unknown"} else "false"

    lines = [
        f"git_commit={commit}",
        f"git_branch={branch}",
        f"git_dirty={dirty}",
    ]
    if status not in {"", "unknown"}:
        lines.append("git_status=")
        lines.extend(status.splitlines())

    Path(output_dir, "git_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return commit


def write_env_info(output_dir: str) -> str:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "PYTORCH_CUDA_ALLOC_CONF",
        "LLAMA_FACTORY_DIR",
    ]
    lines = [f"{key}={os.environ.get(key, '')}" for key in keys]
    env_path = Path(output_dir, "env.txt")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(env_path)


def append_run_index(output_dir: str, row_dict: dict[str, Any]) -> str:
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    index_path = runs_dir / "index.csv"

    fieldnames = list(DEFAULT_INDEX_FIELDS)
    for key in row_dict:
        if key not in fieldnames:
            fieldnames.append(key)

    write_header = not index_path.exists()
    if not write_header:
        with index_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
            if existing:
                fieldnames = existing

    row = {key: row_dict.get(key, "") for key in fieldnames}
    with index_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return str(index_path)


def create_temp_config(config_path: str, run_name: str, output_dir: str, temp_dir: str = ".tmp_configs") -> str:
    src = Path(config_path)
    tmp_dir = Path(temp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f"{run_name}.yaml"

    text = src.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*output_dir\s*:\s*.*$", flags=re.MULTILINE)
    replacement = f"output_dir: {output_dir}"
    if pattern.search(text):
        text = pattern.sub(replacement, text, count=1)
    else:
        text = text.rstrip() + "\n\n" + replacement + "\n"

    target.write_text(text, encoding="utf-8")
    return str(target)


def write_meta(output_dir: str, meta: dict[str, Any]) -> str:
    target = Path(output_dir) / "meta.json"
    target.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(target)


def prepare_run(config_path: str, tag: str | None, launcher: str, ds_config: str | None = None) -> dict[str, str]:
    model_alias = infer_model_alias(config_path)
    method = infer_method(config_path)
    run_name = make_run_name(model_alias=model_alias, method=method, tag=tag)
    output_dir = build_output_dir(model_alias=model_alias, method=method, run_name=run_name)
    launch_time = dt.datetime.now().isoformat(timespec="seconds")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    snapshot_config(config_path, output_dir)
    git_commit = write_git_info(output_dir)
    write_env_info(output_dir)
    temp_config_path = create_temp_config(config_path, run_name, output_dir)

    meta = {
        "run_name": run_name,
        "model_alias": model_alias,
        "method": method,
        "config_path": config_path,
        "output_dir": output_dir,
        "launch_time": launch_time,
        "git_commit": git_commit,
        "launcher": launcher,
        "tag": tag or "",
        "deepspeed_config": ds_config or "",
        "temp_config_path": temp_config_path,
    }
    write_meta(output_dir, meta)
    return meta


def _cmd_prepare(args: argparse.Namespace) -> int:
    meta = prepare_run(
        config_path=args.config_path,
        tag=args.tag,
        launcher=args.launcher,
        ds_config=args.ds_config,
    )
    print(json.dumps(meta, ensure_ascii=False))
    return 0


def _cmd_append_index(args: argparse.Namespace) -> int:
    row = {
        "run_name": args.run_name,
        "model_alias": args.model_alias,
        "method": args.method,
        "config_path": args.config_path,
        "output_dir": args.output_dir,
        "launch_time": args.launch_time,
        "git_commit": args.git_commit,
        "launcher": args.launcher,
        "tag": args.tag,
        "status": args.status,
        "temp_config_path": args.temp_config_path,
        "deepspeed_config": args.ds_config,
    }
    append_run_index(args.output_dir, row)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run manager utilities for experiment tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare run directory and metadata")
    prepare.add_argument("--config-path", required=True)
    prepare.add_argument("--tag", default="")
    prepare.add_argument("--launcher", required=True)
    prepare.add_argument("--ds-config", default="")
    prepare.set_defaults(func=_cmd_prepare)

    append = subparsers.add_parser("append-index", help="Append a row to runs/index.csv")
    append.add_argument("--run-name", required=True)
    append.add_argument("--model-alias", required=True)
    append.add_argument("--method", required=True)
    append.add_argument("--config-path", required=True)
    append.add_argument("--output-dir", required=True)
    append.add_argument("--launch-time", required=True)
    append.add_argument("--git-commit", required=True)
    append.add_argument("--launcher", required=True)
    append.add_argument("--tag", default="")
    append.add_argument("--status", default="unknown")
    append.add_argument("--temp-config-path", default="")
    append.add_argument("--ds-config", default="")
    append.set_defaults(func=_cmd_append_index)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
