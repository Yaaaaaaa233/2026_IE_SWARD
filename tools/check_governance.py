#!/usr/bin/env python3
"""Check shareable repository structure, task records and candidate Git contents.

Python 3.10+, standard library only. This is a guardrail, not a privacy proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

POLICY = "tools/governance_policy.json"
STATES = {"proposed", "ready", "active", "review", "accepted", "blocked", "cancelled"}
TASK_FIELDS = {
    "id", "title", "status", "owner", "actor", "reviewer", "reviewed_at", "branch",
    "base_commit", "write_scope", "dependencies", "acceptance", "evidence", "handoff", "notes",
}


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def snapshot(root: Path, scope: str) -> tuple[dict[str, bytes], list[str]]:
    files: dict[str, bytes] = {}
    errors: list[str] = []
    if scope == "working-tree":
        names = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        entries = [(x.decode("utf-8"), None) for x in names.split(b"\0") if x]
    elif scope == "index":
        entries = []
        for item in git(root, "ls-files", "--stage", "-z").split(b"\0"):
            if not item:
                continue
            metadata, name = item.split(b"\t", 1)
            mode, _, stage = metadata.decode().split()
            path = name.decode("utf-8")
            if stage != "0":
                errors.append(f"Unresolved merge in index: {path}")
                continue
            entries.append((path, mode))
    else:
        entries = []
        for item in git(root, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
            if item:
                metadata, name = item.split(b"\t", 1)
                entries.append((name.decode("utf-8"), metadata.decode().split()[0]))
    for name, mode in entries:
        if name in files:
            continue
        path = root / name
        if mode in {"120000", "160000"} or (scope == "working-tree" and path.is_symlink()):
            errors.append(f"Symlink/submodule requires explicit review: {name}")
            continue
        try:
            if scope == "working-tree":
                if not path.resolve().is_relative_to(root.resolve()):
                    errors.append(f"Path leaves repository: {name}")
                    continue
                files[name] = path.read_bytes()
            else:
                files[name] = git(root, "show", (":" if scope == "index" else "HEAD:") + name)
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot read {name}: {exc}")
    return files, errors


def relative_path(value: str) -> bool:
    return bool(value) and not value.startswith("/") and "\\" not in value and ":" not in value and ".." not in PurePosixPath(value).parts


def exists_in_snapshot(target: str, files: dict[str, bytes]) -> bool:
    return target in files or any(name.startswith(target.rstrip("/") + "/") for name in files)


def outside_fences(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return "\n".join(lines)


def overlaps(a: str, b: str) -> bool:
    a, b = a.rstrip("/"), b.rstrip("/")
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def validate(files: dict[str, bytes]) -> list[str]:
    errors: list[str] = []
    try:
        policy = json.loads(files[POLICY])
        required = policy["required_paths"]
        private_prefixes = policy["private_prefixes"]
        private_files = policy["private_files"]
        forbidden = policy["forbidden_extensions"]
        assets = policy["existing_assets"]
        max_bytes = policy["max_file_bytes"]
    except (KeyError, ValueError, TypeError) as exc:
        return [f"Missing or invalid governance policy: {exc}"]

    for name in required:
        if name not in files:
            errors.append(f"Missing required file: {name}")

    tasks = {}
    for name, content in files.items():
        parts = PurePosixPath(name).parts
        suffix = PurePosixPath(name).suffix.lower()
        basename = PurePosixPath(name).name
        if len(content) > max_bytes:
            errors.append(f"File exceeds reviewed size limit: {name}")
        if name in private_files or any(name.startswith(prefix) for prefix in private_prefixes):
            errors.append(f"Private path cannot be shared: {name}")
        if any(part in {".mimosa", ".ssh", ".venv", "__pycache__"} for part in parts):
            errors.append(f"Local runtime or credentials path: {name}")
        if basename.startswith(".env") and basename != ".env.example":
            errors.append(f"Local environment file: {name}")
        if basename in {".Rhistory", "id_rsa", "id_ed25519", "事件明细.txt"} or basename.startswith("part-"):
            errors.append(f"Local history, key or raw data file: {name}")
        if name in assets:
            if hashlib.sha256(content).hexdigest() != assets[name]["sha256"]:
                errors.append(f"Existing asset changed; publication review needed: {name}")
            continue
        if suffix in forbidden:
            errors.append(f"Data/model/media artifact requires publication review: {name}")
            continue
        if suffix == ".txt" and basename != "requirements.txt":
            errors.append(f"Unreviewed text dump: {name}")
        if b"\0" in content:
            errors.append(f"Unreviewed binary file: {name}")
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"Non-UTF-8 file needs explicit review: {name}")
            continue
        if "\ufffd" in text:
            errors.append(f"Replacement character found: {name}")
        if re.search(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----", text):
            errors.append(f"Private key marker: {name}")
        if name.endswith(".md"):
            for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", outside_fences(text)):
                link = link.strip().strip("<>")
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", link):
                    if not link.startswith(("https://", "http://", "mailto:")):
                        errors.append(f"Nonportable local link: {name} -> {link}")
                    continue
                link = unquote(link.split("#", 1)[0].split("?", 1)[0])
                if not link:
                    continue
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), link))
                if link.startswith("/") or target == ".." or target.startswith("../"):
                    errors.append(f"Link leaves repository: {name} -> {link}")
                elif not exists_in_snapshot(target, files):
                    errors.append(f"Broken or unshared link: {name} -> {link}")
        if name.startswith("docs/tasks/") and name.endswith(".json"):
            try:
                task = json.loads(text)
                if not isinstance(task, dict):
                    raise ValueError("task must be an object")
                if TASK_FIELDS - task.keys():
                    raise ValueError("missing fields: " + ", ".join(sorted(TASK_FIELDS - task.keys())))
                tid = task["id"]
                if not isinstance(tid, str) or not re.fullmatch(r"[A-Z][A-Z0-9]*-\d{3,}", tid):
                    raise ValueError("invalid task id")
                if name != f"docs/tasks/{tid}.json" or tid in tasks:
                    raise ValueError("task filename/id mismatch or duplicate")
                if task["status"] not in STATES:
                    raise ValueError("invalid task state")
                for field in ("write_scope", "dependencies", "acceptance", "evidence"):
                    if not isinstance(task[field], list) or any(not isinstance(x, str) or not x.strip() for x in task[field]):
                        raise ValueError(f"{field} must be an array of nonempty strings")
                for field in ("title", "notes"):
                    if not isinstance(task[field], str):
                        raise ValueError(f"{field} must be text")
                for field in ("owner", "actor", "reviewer", "reviewed_at", "branch", "base_commit", "handoff"):
                    if task[field] is not None and (not isinstance(task[field], str) or not task[field].strip()):
                        raise ValueError(f"{field} must be nonempty text or null")
                for target in task["write_scope"]:
                    if not relative_path(target):
                        raise ValueError("write_scope must contain repository-relative paths")
                if not task["acceptance"]:
                    raise ValueError("acceptance criteria required")
                if task["status"] in {"active", "review", "accepted"}:
                    if not all(task[x] for x in ("actor", "branch", "base_commit", "write_scope")):
                        raise ValueError("execution requires actor, branch, base_commit and write_scope")
                if task["status"] == "accepted" and not all(task[x] for x in ("owner", "reviewer", "reviewed_at", "evidence")):
                    raise ValueError("accepted requires owner, reviewer, reviewed_at and evidence")
                for target in task["evidence"] + ([task["handoff"]] if task["handoff"] else []):
                    if not relative_path(target) or target not in files:
                        raise ValueError(f"missing shared evidence/handoff: {target}")
                tasks[tid] = task
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f"Invalid task {name}: {exc}")

    for tid, task in tasks.items():
        for dependency in task["dependencies"]:
            if dependency not in tasks or dependency == tid:
                errors.append(f"Invalid dependency: {tid} -> {dependency}")
    active = [(tid, task) for tid, task in tasks.items() if task["status"] == "active"]
    for i, (tid, task) in enumerate(active):
        for other_id, other in active[i + 1:]:
            if any(overlaps(a, b) for a in task["write_scope"] for b in other["write_scope"]):
                errors.append(f"Active task write-scope conflict: {tid} and {other_id}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scope", choices=("working-tree", "index", "head"), default="working-tree")
    args = parser.parse_args()
    try:
        files, errors = snapshot(args.root.resolve(), args.scope)
        errors.extend(validate(files))
    except (OSError, ValueError, UnicodeError) as exc:
        errors = [f"Cannot inspect candidate snapshot: {exc}"]
        files = {}
    if errors:
        print(f"Governance FAIL ({args.scope})")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Governance PASS ({args.scope}; {len(files)} files)")
    print("Scope: structure, candidate links, task metadata, known publication guardrails.")
    print("Not a model acceptance or a complete privacy audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
