"""Negative and snapshot checks using synthetic content only."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("governance", ROOT / "tools/check_governance.py")
governance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(governance)


def minimal_files():
    policy = {
        "required_paths": ["README.md", "AGENTS.md"],
        "private_prefixes": ["data/", "artifacts/"],
        "private_files": ["configs/local.toml"],
        "forbidden_extensions": [".csv", ".pdf"],
        "existing_assets": {},
        "max_file_bytes": 10000,
    }
    return {
        governance.POLICY: json.dumps(policy).encode(),
        "README.md": b"# Synthetic test\n[Rules](AGENTS.md)\n",
        "AGENTS.md": b"# Synthetic rules\n",
    }


def task(tid="TEST-001"):
    return {
        "id": tid, "title": "Synthetic task", "status": "active", "owner": None,
        "actor": "synthetic-agent", "reviewer": None, "reviewed_at": None,
        "branch": "work/test", "base_commit": "test-base", "write_scope": ["src/"],
        "dependencies": [], "acceptance": ["synthetic check"], "evidence": [],
        "handoff": None, "notes": "Synthetic test only",
    }


class ContentTests(unittest.TestCase):
    def setUp(self):
        self.files = minimal_files()

    def test_valid_minimal(self):
        self.assertEqual(governance.validate(self.files), [])

    def test_required_file_missing(self):
        del self.files["AGENTS.md"]
        self.assertTrue(any("Missing required" in x for x in governance.validate(self.files)))

    def test_broken_link(self):
        self.files["README.md"] += b"[Missing](missing.md)\n"
        self.assertTrue(any("Broken" in x for x in governance.validate(self.files)))

    def test_code_examples_not_treated_as_links(self):
        self.files["README.md"] += b"```md\n[Example](future.md)\n```\n"
        self.assertEqual(governance.validate(self.files), [])

    def test_private_artifact_rejected_even_if_forced_into_git(self):
        self.files["data/synthetic.csv"] = b"synthetic,value\na,1\n"
        result = governance.validate(self.files)
        self.assertTrue(any("Private path" in x for x in result))
        self.assertTrue(any("publication review" in x for x in result))

    def test_machine_config_rejected(self):
        self.files["configs/local.toml"] = b"synthetic = true\n"
        self.assertTrue(any("Private path" in x for x in governance.validate(self.files)))

    def test_secret_marker_rejected(self):
        self.files["example.md"] = ("-----BEGIN " + "OPENSSH " + "PRIVATE KEY-----").encode()
        self.assertTrue(any("Private key marker" in x for x in governance.validate(self.files)))

    def test_accepted_without_review_rejected(self):
        record = task()
        record["status"] = "accepted"
        self.files["docs/tasks/TEST-001.json"] = json.dumps(record).encode()
        self.assertTrue(any("accepted requires" in x for x in governance.validate(self.files)))

    def test_accepted_with_real_record_fields_allowed(self):
        record = task()
        record.update(status="accepted", owner="synthetic-owner", reviewer="synthetic-reviewer",
                      reviewed_at="2026-09-16", evidence=["docs/check.md"])
        self.files["docs/check.md"] = b"# Synthetic evidence\n"
        self.files["docs/tasks/TEST-001.json"] = json.dumps(record).encode()
        self.assertEqual(governance.validate(self.files), [])

    def test_active_overlap_rejected(self):
        first, second = task(), task("TEST-002")
        second["write_scope"] = ["src/features.py"]
        for record in (first, second):
            self.files[f'docs/tasks/{record["id"]}.json'] = json.dumps(record).encode()
        self.assertTrue(any("write-scope conflict" in x for x in governance.validate(self.files)))

    def test_disjoint_active_tasks_allowed(self):
        first, second = task(), task("TEST-002")
        second["write_scope"] = ["docs/report.md"]
        for record in (first, second):
            self.files[f'docs/tasks/{record["id"]}.json'] = json.dumps(record).encode()
        self.assertEqual(governance.validate(self.files), [])

    def test_asset_fingerprint_change_rejected(self):
        policy = json.loads(self.files[governance.POLICY])
        policy["existing_assets"]["existing.pdf"] = {"sha256": "0" * 64}
        self.files[governance.POLICY] = json.dumps(policy).encode()
        self.files["existing.pdf"] = b"synthetic asset"
        self.assertTrue(any("Existing asset changed" in x for x in governance.validate(self.files)))

    def test_absolute_link_rejected(self):
        self.files["README.md"] += b"[Local](/some/device/file.md)\n"
        self.assertTrue(any("leaves repository" in x for x in governance.validate(self.files)))


class SnapshotTests(unittest.TestCase):
    def test_head_uses_committed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            for name, content in minimal_files().items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Synthetic Test",
                            "-c", "user.email=synthetic@example.invalid", "-c", "commit.gpgsign=false",
                            "commit", "-qm", "Synthetic fixture"], check=True)
            (root / "README.md").write_text("[Broken](missing.md)\n", encoding="utf-8")
            committed, errors = governance.snapshot(root, "head")
            self.assertEqual(errors + governance.validate(committed), [])
            working, _ = governance.snapshot(root, "working-tree")
            self.assertTrue(any("Broken" in x for x in governance.validate(working)))

    def test_cli_returns_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            result = subprocess.run([sys.executable, str(ROOT / "tools/check_governance.py"),
                                     "--root", str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Governance FAIL", result.stdout)
            self.assertNotIn("Traceback", result.stderr)

    def test_index_uses_staged_bytes_not_repaired_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            files = minimal_files()
            files["README.md"] += b"[Broken](missing.md)\n"
            for name, content in files.items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            (root / "README.md").write_bytes(minimal_files()["README.md"])
            staged, errors = governance.snapshot(root, "index")
            self.assertEqual(errors, [])
            self.assertTrue(any("Broken" in x for x in governance.validate(staged)))
            working, errors = governance.snapshot(root, "working-tree")
            self.assertEqual(errors + governance.validate(working), [])

    def test_ignored_private_file_not_read_until_force_added(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            for name, content in minimal_files().items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content)
            (root / ".gitignore").write_text("data/\n", encoding="utf-8")
            (root / "data").mkdir()
            (root / "data/synthetic.csv").write_text("synthetic\n", encoding="utf-8")
            working, errors = governance.snapshot(root, "working-tree")
            self.assertNotIn("data/synthetic.csv", working)
            self.assertEqual(errors + governance.validate(working), [])
            subprocess.run(["git", "-C", str(root), "add", "-f", "data/synthetic.csv"], check=True)
            working, _ = governance.snapshot(root, "working-tree")
            self.assertTrue(any("Private path" in x for x in governance.validate(working)))


if __name__ == "__main__":
    unittest.main()
