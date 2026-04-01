"""ClearML wrapper: create (or clone) a task and enqueue it.

Usage:
    uv run python scripts/clearml_run.py                                    # 기본 실행
    uv run python scripts/clearml_run.py trainer.max_epochs=5 logger=csv    # Hydra 인자 전달
    uv run python scripts/clearml_run.py --clone <task_id>                  # 기존 태스크 복제 후 enqueue
    uv run python scripts/clearml_run.py --docker clearml-worker:latest     # Docker 이미지 지정
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from clearml import Task


def parse_packages_from_pyproject(pyproject_path: str | Path) -> list[str]:
    """pyproject.toml에서 [project] dependencies를 파싱하여 패키지 목록을 반환한다.

    주석(#)과 빈 줄을 제거하고, 개발 전용 패키지(pre-commit, pytest)는 제외한다.
    """
    text = Path(pyproject_path).read_text()

    match = re.search(
        r"dependencies\s*=\s*\[(.*?)\]",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"dependencies not found in {pyproject_path}")

    raw_lines = match.group(1).split("\n")

    dev_only = {"pre-commit", "pytest"}
    packages: list[str] = []
    for line in raw_lines:
        line = line.split("#")[0].strip().strip(",").strip('"').strip("'")
        if not line:
            continue
        pkg_name = re.split(r"[><=!~\[]", line)[0].strip().lower()
        if pkg_name in dev_only:
            continue
        packages.append(line)

    return packages


def create_task(
    *,
    project_name: str,
    task_name: str,
    script: str,
    packages: list[str],
    docker: str | None,
    hydra_args: list[str],
) -> Task:
    task = Task.create(
        project_name=project_name,
        task_name=task_name,
        repo=".",
        script=script,
        working_directory=".",
        packages=packages,
        docker=docker,
    )
    if hydra_args:
        params: dict[str, str] = {}
        for arg in hydra_args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                params[f"Args/{key}"] = value
            else:
                params[f"Args/{arg}"] = ""
        task.set_parameters(params)
    return task


def clone_task(base_task_id: str, hydra_args: list[str]) -> Task:
    task = Task.clone(source_task=base_task_id)
    if hydra_args:
        params: dict[str, str] = {}
        for arg in hydra_args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                params[f"Args/{key}"] = value
            else:
                params[f"Args/{arg}"] = ""
        task.set_parameters(params)
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description="ClearML task launcher")
    parser.add_argument(
        "--project",
        default="ClearML-Learning",
        help="ClearML project name (default: ClearML-Learning)",
    )
    parser.add_argument(
        "--name",
        default="train",
        help="Task name (default: train)",
    )
    parser.add_argument(
        "--script",
        default="scripts/clearml_train.py",
        help="Entry point script (default: scripts/clearml_train.py)",
    )
    parser.add_argument(
        "--queue",
        default="default",
        help="ClearML queue name (default: default)",
    )
    parser.add_argument(
        "--clone",
        default=None,
        metavar="TASK_ID",
        help="Clone an existing task instead of creating a new one",
    )
    parser.add_argument(
        "--docker",
        default=None,
        help="Docker image for the agent (e.g. clearml-worker:latest)",
    )

    args, hydra_args = parser.parse_known_args()

    project_root = Path(__file__).resolve().parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if args.clone:
        print(f"Cloning task {args.clone} ...")
        task = clone_task(args.clone, hydra_args)
    else:
        packages = parse_packages_from_pyproject(pyproject_path)
        print(f"Parsed {len(packages)} packages from {pyproject_path}:")
        for pkg in packages:
            print(f"  - {pkg}")

        task = create_task(
            project_name=args.project,
            task_name=args.name,
            script=args.script,
            packages=packages,
            docker=args.docker,
            hydra_args=hydra_args,
        )

    Task.enqueue(task=task, queue_name=args.queue)
    print(f"\nTask enqueued: {task.id}")
    print(f"  project : {args.project}")
    print(f"  queue   : {args.queue}")


if __name__ == "__main__":
    main()
