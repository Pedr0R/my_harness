from __future__ import annotations

import pathlib
import subprocess
import tempfile
import time
import unittest

from workspace import CheckpointState, WorkspaceCheckpoint


def _git(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )


def _init_git_repo(repo: pathlib.Path, *, commit_initial: bool = True) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    if commit_initial:
        (repo / "tracked.txt").write_text("initial content\n", encoding="utf-8")
        (repo / "to_delete.txt").write_text("keep me\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "initial commit")


class TestWorkspaceCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_clean_repo_rollback_restores_modified_deleted_and_removes_new_files(self) -> None:
        repo = self.root / "clean_repo"
        _init_git_repo(repo)

        cp = WorkspaceCheckpoint(root_dir=repo)
        self.assertTrue(cp.snapshot())
        self.assertEqual(cp.state, CheckpointState.ACTIVE)
        self.assertIsNotNone(cp.snapshot_tree_sha)

        (repo / "tracked.txt").write_text("mutated by agent\n", encoding="utf-8")
        (repo / "to_delete.txt").unlink()
        nested = repo / "created_dir" / "sub" / "new_file.txt"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("temporary artifact\n", encoding="utf-8")

        result = cp.rollback()
        self.assertTrue(result.ok)
        self.assertTrue(result.active)
        self.assertEqual(result.removed_files, 1)
        self.assertEqual(cp.state, CheckpointState.ROLLED_BACK)

        self.assertEqual((repo / "tracked.txt").read_text(encoding="utf-8"), "initial content\n")
        self.assertEqual((repo / "to_delete.txt").read_text(encoding="utf-8"), "keep me\n")
        self.assertFalse(nested.exists())
        self.assertFalse((repo / "created_dir").exists())
        self.assertEqual(_git(repo, "status", "--porcelain").stdout.strip(), "")
        self.assertEqual(_git(repo, "stash", "list").stdout.strip(), "")

    def test_dirty_repo_preserves_preexisting_unstaged_and_untracked_files(self) -> None:
        repo = self.root / "dirty_repo"
        _init_git_repo(repo)

        (repo / "tracked.txt").write_text("preexisting dirty edit\n", encoding="utf-8")
        (repo / "pre_untracked.txt").write_text("preexisting untracked\n", encoding="utf-8")

        cp = WorkspaceCheckpoint(root_dir=repo)
        self.assertTrue(cp.snapshot())

        (repo / "tracked.txt").write_text("agent overwrote dirty edit\n", encoding="utf-8")
        (repo / "pre_untracked.txt").write_text("agent corrupted untracked\n", encoding="utf-8")
        (repo / "agent_created.txt").write_text("should be removed\n", encoding="utf-8")

        result = cp.rollback()
        self.assertTrue(result.ok)
        self.assertEqual(result.removed_files, 1)

        self.assertEqual((repo / "tracked.txt").read_text(encoding="utf-8"), "preexisting dirty edit\n")
        self.assertEqual((repo / "pre_untracked.txt").read_text(encoding="utf-8"), "preexisting untracked\n")
        self.assertFalse((repo / "agent_created.txt").exists())

        status_lines = _git(repo, "status", "--porcelain").stdout.splitlines()
        self.assertIn(" M tracked.txt", status_lines)
        self.assertIn("?? pre_untracked.txt", status_lines)
        self.assertEqual(len(status_lines), 2)

    def test_preexisting_staged_changes_preserved_across_snapshot_and_rollback(self) -> None:
        repo = self.root / "staged_repo"
        _init_git_repo(repo)

        (repo / "tracked.txt").write_text("staged v2\n", encoding="utf-8")
        _git(repo, "add", "tracked.txt")
        (repo / "tracked.txt").write_text("unstaged v3\n", encoding="utf-8")
        expected_cached_diff = _git(repo, "diff", "--cached").stdout

        cp = WorkspaceCheckpoint(root_dir=repo)
        self.assertTrue(cp.snapshot())
        self.assertEqual(_git(repo, "diff", "--cached").stdout, expected_cached_diff)

        (repo / "tracked.txt").write_text("agent broke it\n", encoding="utf-8")
        (repo / "extra.txt").write_text("staged by agent\n", encoding="utf-8")
        _git(repo, "add", "tracked.txt", "extra.txt")

        result = cp.rollback()
        self.assertTrue(result.ok)
        self.assertEqual((repo / "tracked.txt").read_text(encoding="utf-8"), "unstaged v3\n")
        self.assertFalse((repo / "extra.txt").exists())
        self.assertEqual(_git(repo, "diff", "--cached").stdout, expected_cached_diff)

    def test_unborn_head_repo_snapshot_and_rollback(self) -> None:
        repo = self.root / "unborn_repo"
        _init_git_repo(repo, commit_initial=False)
        (repo / "draft.txt").write_text("first draft\n", encoding="utf-8")

        cp = WorkspaceCheckpoint(root_dir=repo)
        self.assertTrue(cp.snapshot())

        (repo / "draft.txt").write_text("broken draft\n", encoding="utf-8")
        (repo / "junk.txt").write_text("junk\n", encoding="utf-8")
        _git(repo, "add", "draft.txt", "junk.txt")

        result = cp.rollback()
        self.assertTrue(result.ok)
        self.assertEqual((repo / "draft.txt").read_text(encoding="utf-8"), "first draft\n")
        self.assertFalse((repo / "junk.txt").exists())
        self.assertEqual(_git(repo, "diff", "--cached").stdout, "")

    def test_non_git_directory_degrades_gracefully(self) -> None:
        non_git = self.root / "plain_dir"
        non_git.mkdir()
        (non_git / "note.txt").write_text("hello", encoding="utf-8")

        cp = WorkspaceCheckpoint(root_dir=non_git)
        self.assertFalse(cp.snapshot())
        self.assertEqual(cp.state, CheckpointState.UNAVAILABLE)

        result = cp.rollback()
        self.assertFalse(result.ok)
        self.assertFalse(result.active)
        self.assertEqual((non_git / "note.txt").read_text(encoding="utf-8"), "hello")

    def test_sessions_directory_is_never_deleted_on_rollback(self) -> None:
        for gitignore_sessions in (True, False):
            with self.subTest(gitignore_sessions=gitignore_sessions):
                repo = self.root / f"sessions_repo_{gitignore_sessions}"
                _init_git_repo(repo)
                if gitignore_sessions:
                    (repo / ".gitignore").write_text(".sessions/\n", encoding="utf-8")
                    _git(repo, "add", ".gitignore")
                    _git(repo, "commit", "-m", "ignore sessions")

                sessions_dir = repo / ".sessions"
                sessions_dir.mkdir()
                existing_session = sessions_dir / "session_existing.json"
                existing_session.write_text('{"id": "1"}', encoding="utf-8")

                cp = WorkspaceCheckpoint(root_dir=repo)
                self.assertTrue(cp.snapshot())

                new_session = sessions_dir / "session_new.json"
                new_session.write_text('{"id": "2"}', encoding="utf-8")
                (repo / "tracked.txt").write_text("corrupted\n", encoding="utf-8")
                (repo / "temp.txt").write_text("remove me\n", encoding="utf-8")

                result = cp.rollback()
                self.assertTrue(result.ok)
                self.assertEqual((repo / "tracked.txt").read_text(encoding="utf-8"), "initial content\n")
                self.assertFalse((repo / "temp.txt").exists())
                self.assertEqual(existing_session.read_text(encoding="utf-8"), '{"id": "1"}')
                self.assertEqual(new_session.read_text(encoding="utf-8"), '{"id": "2"}')

    def test_rollback_is_idempotent_and_works_after_commit(self) -> None:
        repo = self.root / "idempotent_repo"
        _init_git_repo(repo)

        cp = WorkspaceCheckpoint(root_dir=repo)
        self.assertTrue(cp.snapshot())

        (repo / "tracked.txt").write_text("turn edit\n", encoding="utf-8")
        (repo / "added.txt").write_text("added\n", encoding="utf-8")
        commit_res = cp.commit()
        self.assertTrue(commit_res.ok)
        self.assertEqual(cp.state, CheckpointState.COMMITTED)

        first = cp.rollback()
        second = cp.rollback()

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(first.removed_files, 1)
        self.assertEqual(second.removed_files, 0)
        self.assertEqual((repo / "tracked.txt").read_text(encoding="utf-8"), "initial content\n")
        self.assertFalse((repo / "added.txt").exists())
        self.assertEqual(_git(repo, "status", "--porcelain").stdout.strip(), "")

    def test_context_manager_commits_on_success_and_rolls_back_on_error(self) -> None:
        repo = self.root / "ctx_repo"
        _init_git_repo(repo)

        with WorkspaceCheckpoint(root_dir=repo) as cp_ok:
            (repo / "tracked.txt").write_text("committed change\n", encoding="utf-8")
        self.assertEqual(cp_ok.state, CheckpointState.COMMITTED)
        self.assertEqual((repo / "tracked.txt").read_text(encoding="utf-8"), "committed change\n")

        with self.assertRaises(RuntimeError):
            with WorkspaceCheckpoint(root_dir=repo) as cp_err:
                (repo / "tracked.txt").write_text("failed change\n", encoding="utf-8")
                raise RuntimeError("boom")
        self.assertEqual(cp_err.state, CheckpointState.ROLLED_BACK)
        self.assertEqual((repo / "tracked.txt").read_text(encoding="utf-8"), "committed change\n")

    def test_snapshot_overhead_in_50_file_repo_under_250ms_warm(self) -> None:
        repo = self.root / "perf_repo"
        _init_git_repo(repo, commit_initial=False)
        for i in range(50):
            (repo / f"module_{i:02d}.py").write_text(f"VALUE = {i}\n" * 10, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-m", "add 50 files")

        # Warmup to exclude cold process/cache startup from measurement
        warmup = WorkspaceCheckpoint(root_dir=repo)
        self.assertTrue(warmup.snapshot())

        timings_ms: list[float] = []
        for _ in range(3):
            cp = WorkspaceCheckpoint(root_dir=repo)
            t0 = time.perf_counter()
            ok = cp.snapshot()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.assertTrue(ok)
            timings_ms.append(elapsed_ms)

        median_ms = sorted(timings_ms)[len(timings_ms) // 2]
        self.assertLess(
            median_ms,
            250.0,
            f"Warm snapshot median {median_ms:.1f}ms exceeded 250ms budget (samples={timings_ms})",
        )


if __name__ == "__main__":
    unittest.main()
