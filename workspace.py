from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import functools
import os
import pathlib
import shutil
import subprocess
import tempfile
from types import TracebackType


class CheckpointState(str, Enum):
    """Estados explícitos do ciclo de vida de um checkpoint de workspace."""

    IDLE = "idle"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CheckpointResult:
    """Resultado estruturado de operações de confirmação ou rollback."""

    ok: bool
    active: bool
    message: str
    restored_files: int = 0
    removed_files: int = 0


def _normalize_excluded_dirs(excluded_dirs: frozenset[str]) -> frozenset[str]:
    normalized: set[str] = set()
    for item in excluded_dirs:
        cleaned = item.replace("\\", "/").strip("/")
        if cleaned:
            normalized.add(cleaned)
    return frozenset(normalized)


@functools.lru_cache(maxsize=1)
def _resolve_git_executable() -> str:
    """Resolve o binário Git direto no Windows (mingw64/bin/git.exe) para evitar o overhead do wrapper cmd/git.exe."""
    git_path = shutil.which("git")
    if not git_path:
        return "git"
    if os.name == "nt":
        resolved = pathlib.Path(git_path).resolve()
        if resolved.parent.name.lower() == "cmd":
            direct = resolved.parent.parent / "mingw64" / "bin" / "git.exe"
            if direct.is_file():
                return str(direct)
    return git_path


def _find_git_repo(start_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path] | None:
    """Localiza a raiz do repositório Git e o diretório .git sem iniciar subprocessos."""
    try:
        resolved = start_dir.resolve()
    except OSError:
        return None
    if not resolved.is_dir():
        return None

    for candidate in (resolved, *resolved.parents):
        dot_git = candidate / ".git"
        if dot_git.is_dir():
            if (dot_git / "HEAD").is_file():
                return candidate, dot_git
            continue
        if dot_git.is_file():
            try:
                first_line = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
            except (OSError, IndexError):
                continue
            if first_line.lower().startswith("gitdir:"):
                raw_target = first_line.split(":", 1)[1].strip()
                git_dir = pathlib.Path(raw_target)
                if not git_dir.is_absolute():
                    git_dir = (candidate / git_dir).resolve()
                else:
                    git_dir = git_dir.resolve()
                if (git_dir / "HEAD").is_file():
                    return candidate, git_dir
    return None


def _is_excluded_path(rel_path: str, excluded_dirs: frozenset[str]) -> bool:
    parts = pathlib.PurePosixPath(rel_path).parts
    if not parts:
        return False
    if parts[0] == ".git":
        return True
    for excluded in excluded_dirs:
        if "/" in excluded:
            if rel_path == excluded or rel_path.startswith(f"{excluded}/"):
                return True
        elif excluded == parts[0] or excluded in parts[:-1]:
            return True
    return False


def _prune_empty_parents(
    file_path: pathlib.Path,
    stop_root: pathlib.Path,
    excluded_dirs: frozenset[str],
) -> None:
    parent = file_path.parent
    while parent != stop_root and parent.is_relative_to(stop_root):
        if parent.name == ".git" or parent.name in excluded_dirs:
            break
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


class WorkspaceCheckpoint:
    """Snapshot e rollback idempotente de workspace usando Git tree plumbing com GIT_INDEX_FILE temporário."""

    def __init__(
        self,
        root_dir: str | pathlib.Path | None = None,
        excluded_dirs: frozenset[str] = frozenset({".sessions"}),
    ) -> None:
        self._root_dir: pathlib.Path | None = pathlib.Path(root_dir) if root_dir is not None else None
        self.excluded_dirs: frozenset[str] = _normalize_excluded_dirs(frozenset(excluded_dirs))
        self.state: CheckpointState = CheckpointState.IDLE
        self.repo_root: pathlib.Path | None = None
        self._git_dir: pathlib.Path | None = None
        self.snapshot_tree_sha: str | None = None
        self._snapshot_files: frozenset[str] | None = None
        self._original_index_bytes: bytes | None = None
        self._original_index_tree: str | None = None

    @property
    def is_active(self) -> bool:
        return self.state is CheckpointState.ACTIVE

    @property
    def snapshot_files(self) -> frozenset[str]:
        if self._snapshot_files is not None:
            return self._snapshot_files
        if self.snapshot_tree_sha is None or self.repo_root is None:
            return frozenset()
        res = self._run_git(["ls-tree", "-r", "--name-only", "-z", self.snapshot_tree_sha])
        if res.returncode != 0:
            return frozenset()
        files = frozenset(p for p in res.stdout.decode("utf-8").split("\0") if p)
        self._snapshot_files = files
        return files

    @property
    def original_index_tree(self) -> str | None:
        if self._original_index_tree is not None:
            return self._original_index_tree
        if self._original_index_bytes is None or self.repo_root is None:
            return None
        with tempfile.TemporaryDirectory(prefix="my_harness_idx_") as tmp_dir:
            orig_index = pathlib.Path(tmp_dir) / "index"
            orig_index.write_bytes(self._original_index_bytes)
            res = self._run_git(["write-tree"], index_file=orig_index)
            if res.returncode != 0:
                return None
            tree_sha = res.stdout.decode("utf-8").strip()
            self._original_index_tree = tree_sha or None
            return self._original_index_tree

    def _run_git(
        self,
        args: list[str],
        *,
        index_file: pathlib.Path | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        env: dict[str, str] | None = None
        if index_file is not None:
            env = os.environ.copy()
            env["GIT_INDEX_FILE"] = str(index_file)
        return subprocess.run(
            [_resolve_git_executable(), *args],
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            check=False,
        )

    def _exclude_pathspecs(self) -> list[str]:
        return [f":(exclude,glob)**/{d}/**" for d in sorted(self.excluded_dirs)]

    def snapshot(self) -> bool:
        """Captura uma árvore Git imutável do estado atual do workspace sem alterar .git/index ou git stash."""
        start_dir = self._root_dir if self._root_dir is not None else pathlib.Path.cwd()
        found = _find_git_repo(start_dir)
        if found is None:
            self.state = CheckpointState.UNAVAILABLE
            return False

        self.repo_root, self._git_dir = found
        self._snapshot_files = None
        self._original_index_tree = None

        real_index = self._git_dir / "index"
        try:
            self._original_index_bytes = real_index.read_bytes() if real_index.is_file() else None
        except OSError:
            self.state = CheckpointState.UNAVAILABLE
            return False

        try:
            with tempfile.TemporaryDirectory(prefix="my_harness_cp_") as tmp_dir:
                temp_index = pathlib.Path(tmp_dir) / "index"
                if self._original_index_bytes is not None:
                    temp_index.write_bytes(self._original_index_bytes)

                add_cmd = ["add", "-A", "--", ".", *self._exclude_pathspecs()]
                add_res = self._run_git(add_cmd, index_file=temp_index)
                if add_res.returncode != 0:
                    self.state = CheckpointState.UNAVAILABLE
                    return False

                tree_res = self._run_git(["write-tree"], index_file=temp_index)
                if tree_res.returncode != 0:
                    self.state = CheckpointState.UNAVAILABLE
                    return False

                tree_sha = tree_res.stdout.decode("utf-8").strip()
                if not tree_sha:
                    self.state = CheckpointState.UNAVAILABLE
                    return False
        except (FileNotFoundError, OSError):
            self.state = CheckpointState.UNAVAILABLE
            return False

        self.snapshot_tree_sha = tree_sha
        self.state = CheckpointState.ACTIVE
        return True

    def commit(self) -> CheckpointResult:
        """Marca o checkpoint como concluído preservando o snapshot para /undo manual."""
        if self.snapshot_tree_sha is None:
            return CheckpointResult(
                ok=False,
                active=False,
                message="Nenhum checkpoint ativo.",
            )
        self.state = CheckpointState.COMMITTED
        return CheckpointResult(
            ok=True,
            active=True,
            message=f"Checkpoint confirmado ({self.snapshot_tree_sha[:8]}).",
        )

    def rollback(self) -> CheckpointResult:
        """Restaura de forma idempotente arquivos do workspace e o índice Git para o snapshot capturado."""
        if self.snapshot_tree_sha is None or self.repo_root is None or self._git_dir is None:
            return CheckpointResult(
                ok=False,
                active=False,
                message="Nenhum checkpoint ativo.",
            )

        try:
            expected_files = self.snapshot_files
            ls_res = self._run_git(["ls-files", "--cached", "--others", "--exclude-standard", "-z"])
            if ls_res.returncode != 0:
                stderr = ls_res.stderr.decode("utf-8", errors="replace").strip()
                return CheckpointResult(
                    ok=False,
                    active=True,
                    message=f"Falha ao listar arquivos do workspace: {stderr}",
                )

            current_files = [p for p in ls_res.stdout.decode("utf-8").split("\0") if p]
            removed_files = 0
            for rel_path in current_files:
                if rel_path in expected_files or _is_excluded_path(rel_path, self.excluded_dirs):
                    continue
                target = self.repo_root / pathlib.PurePosixPath(rel_path)
                if not target.is_relative_to(self.repo_root):
                    continue
                if target.is_symlink() or target.is_file():
                    target.unlink()
                    removed_files += 1
                    _prune_empty_parents(target, self.repo_root, self.excluded_dirs)

            with tempfile.TemporaryDirectory(prefix="my_harness_rb_") as tmp_dir:
                temp_index = pathlib.Path(tmp_dir) / "index"
                read_res = self._run_git(["read-tree", self.snapshot_tree_sha], index_file=temp_index)
                if read_res.returncode != 0:
                    stderr = read_res.stderr.decode("utf-8", errors="replace").strip()
                    return CheckpointResult(
                        ok=False,
                        active=True,
                        message=f"Falha ao carregar árvore do checkpoint: {stderr}",
                    )

                checkout_res = self._run_git(["checkout-index", "-a", "-f"], index_file=temp_index)
                if checkout_res.returncode != 0:
                    stderr = checkout_res.stderr.decode("utf-8", errors="replace").strip()
                    return CheckpointResult(
                        ok=False,
                        active=True,
                        message=f"Falha ao restaurar arquivos do checkpoint: {stderr}",
                    )

                if self._original_index_bytes is not None:
                    orig_tree = self._original_index_tree
                    if orig_tree is None:
                        orig_index = pathlib.Path(tmp_dir) / "orig_index"
                        orig_index.write_bytes(self._original_index_bytes)
                        wt_res = self._run_git(["write-tree"], index_file=orig_index)
                        if wt_res.returncode != 0:
                            stderr = wt_res.stderr.decode("utf-8", errors="replace").strip()
                            return CheckpointResult(
                                ok=False,
                                active=True,
                                message=f"Falha ao reconstruir índice original: {stderr}",
                            )
                        orig_tree = wt_res.stdout.decode("utf-8").strip()
                        self._original_index_tree = orig_tree

                    restore_idx = self._run_git(["read-tree", orig_tree])
                    if restore_idx.returncode != 0:
                        stderr = restore_idx.stderr.decode("utf-8", errors="replace").strip()
                        return CheckpointResult(
                            ok=False,
                            active=True,
                            message=f"Falha ao restaurar índice Git: {stderr}",
                        )
                elif (self._git_dir / "index").exists():
                    empty_idx = self._run_git(["read-tree", "--empty"])
                    if empty_idx.returncode != 0:
                        stderr = empty_idx.stderr.decode("utf-8", errors="replace").strip()
                        return CheckpointResult(
                            ok=False,
                            active=True,
                            message=f"Falha ao limpar índice Git: {stderr}",
                        )
        except (FileNotFoundError, OSError) as exc:
            return CheckpointResult(
                ok=False,
                active=True,
                message=f"Erro ao executar rollback do workspace: {exc}",
            )

        self.state = CheckpointState.ROLLED_BACK
        restored_files = len(expected_files)
        return CheckpointResult(
            ok=True,
            active=True,
            message=(
                f"Workspace restaurado para o checkpoint {self.snapshot_tree_sha[:8]} "
                f"({restored_files} arquivo(s) restaurado(s), {removed_files} removido(s))."
            ),
            restored_files=restored_files,
            removed_files=removed_files,
        )

    def __enter__(self) -> WorkspaceCheckpoint:
        self.snapshot()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if self.is_active:
            if exc_type is not None:
                self.rollback()
            else:
                self.commit()
        return False
