from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

from agent import discover_context
from tools import (
    APPROVAL_REQUIRED_TOOLS,
    TOOLS,
    TOOL_SCHEMAS,
    _truncate,
    build_tools,
    tool_bash,
    tool_bash_async,
    tool_find_files,
    tool_grep,
    tool_list_dir,
    tool_read_file,
    tool_search_replace,
    tool_write_file,
)


def _create_escaping_link(
    root: pathlib.Path,
    outside: pathlib.Path,
    outside_file: pathlib.Path,
    link_name: str,
    dir_name: str,
) -> str | None:
    link_path = root / link_name
    try:
        os.symlink(outside_file, link_path)
        return link_name
    except OSError:
        if os.name == "nt":
            junction = root / dir_name
            proc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                return dir_name
        return None


class TestTruncate(unittest.TestCase):
    def test_returns_text_unchanged_below_and_at_limit(self) -> None:
        self.assertEqual(_truncate("short", 10), "short")
        self.assertEqual(_truncate("0123456789", 10), "0123456789")

    def test_truncates_and_appends_marker_above_limit(self) -> None:
        self.assertEqual(_truncate("0123456789abcdef", 10), "0123456789\n[truncado]")


class TestFileTools(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_write_file_creates_parents_writes_utf8_and_overwrites(self) -> None:
        target = self.root / "nested" / "deep" / "notes.txt"
        first_content = "olá mundo — ação"

        res1 = tool_write_file(str(target), first_content)
        self.assertEqual(res1, f"escreveu {len(first_content)} bytes")
        self.assertEqual(target.read_text(encoding="utf-8"), first_content)

        second_content = "updated"
        res2 = tool_write_file(str(target), second_content)
        self.assertEqual(res2, f"escreveu {len(second_content)} bytes")
        self.assertEqual(target.read_text(encoding="utf-8"), second_content)

    def test_read_file_line_numbering_and_pagination(self) -> None:
        target = self.root / "lines.txt"
        target.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n", encoding="utf-8")

        full = tool_read_file(str(target))
        self.assertEqual(
            full,
            "   1: alpha\n   2: beta\n   3: gamma\n   4: delta\n   5: epsilon",
        )

        page = tool_read_file(str(target), offset=1, limit=2)
        self.assertEqual(page, "   1: beta\n   2: gamma")

    def test_read_file_replaces_invalid_utf8_bytes(self) -> None:
        target = self.root / "binary.txt"
        target.write_bytes(b"ok\xff\xfebad")

        out = tool_read_file(str(target))
        self.assertIn("ok", out)
        self.assertIn("bad", out)
        self.assertIn("\ufffd", out)

    def test_read_file_truncates_large_output(self) -> None:
        target = self.root / "large.txt"
        target.write_text(("x" * 200 + "\n") * 200, encoding="utf-8")

        out = tool_read_file(str(target))
        self.assertTrue(out.endswith("\n[truncado]"))
        self.assertEqual(len(out), 25_000 + len("\n[truncado]"))

    def test_read_file_missing_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            tool_read_file(str(self.root / "does_not_exist.txt"))

    def test_search_replace_unique_match_succeeds(self) -> None:
        target = self.root / "code.py"
        target.write_text("x = 1\ny = 2\n", encoding="utf-8")

        res = tool_search_replace(str(target), "x = 1", "x = 42")
        self.assertEqual(res, "OK")
        self.assertEqual(target.read_text(encoding="utf-8"), "x = 42\ny = 2\n")

    def test_search_replace_zero_matches_leaves_file_unchanged(self) -> None:
        target = self.root / "code.py"
        original = "x = 1\ny = 2\n"
        target.write_text(original, encoding="utf-8")

        res = tool_search_replace(str(target), "z = 3", "z = 9")
        self.assertTrue(res.startswith("ERRO:"))
        self.assertIn("0x", res)
        self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_search_replace_multiple_matches_leaves_file_unchanged(self) -> None:
        target = self.root / "dup.txt"
        original = "token\nother\ntoken\n"
        target.write_text(original, encoding="utf-8")

        res = tool_search_replace(str(target), "token", "replaced")
        self.assertTrue(res.startswith("ERRO:"))
        self.assertIn("2x", res)
        self.assertEqual(target.read_text(encoding="utf-8"), original)


class TestToolGrep(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_literal_search_across_nested_files_and_single_file(self) -> None:
        (self.root / "alpha.py").write_text("first\nhello world [literal]\nlast\n", encoding="utf-8")
        nested = self.root / "pkg" / "beta.txt"
        nested.parent.mkdir(parents=True)
        nested.write_text("hello world [literal] again\nother\n", encoding="utf-8")
        (self.root / "unused.txt").write_text("nothing here\n", encoding="utf-8")

        dir_out = tool_grep("hello world [literal]", path=str(self.root))
        self.assertEqual(
            dir_out.splitlines(),
            [
                "alpha.py:2: hello world [literal]",
                "pkg/beta.txt:1: hello world [literal] again",
            ],
        )

        file_out = tool_grep("hello world [literal]", path=str(nested))
        self.assertEqual(file_out, f"{nested.as_posix()}:1: hello world [literal] again")

    def test_regex_search_and_invalid_regex_error(self) -> None:
        target = self.root / "funcs.py"
        target.write_text("def calc_1():\n    pass\ndef other():\ndef calc_42():\n", encoding="utf-8")

        out = tool_grep(r"def\s+calc_\d+", path=str(self.root), regex=True)
        self.assertEqual(
            out.splitlines(),
            [
                "funcs.py:1: def calc_1():",
                "funcs.py:4: def calc_42():",
            ],
        )

        err = tool_grep("[unclosed", path=str(self.root), regex=True)
        self.assertTrue(err.startswith("ERRO: expressão regular inválida:"))

    def test_file_pattern_filters_directory_walk(self) -> None:
        (self.root / "app.py").write_text("TARGET_TOKEN = 1\n", encoding="utf-8")
        (self.root / "notes.md").write_text("TARGET_TOKEN in docs\n", encoding="utf-8")
        sub = self.root / "sub"
        sub.mkdir()
        (sub / "helper.py").write_text("return TARGET_TOKEN\n", encoding="utf-8")

        out = tool_grep("TARGET_TOKEN", path=str(self.root), file_pattern="*.py")
        self.assertEqual(
            out.splitlines(),
            [
                "app.py:1: TARGET_TOKEN = 1",
                "sub/helper.py:1: return TARGET_TOKEN",
            ],
        )

    def test_ignores_excluded_dirs_sensitive_files_and_binary_files(self) -> None:
        ignored_files = [
            ".git/config",
            "__pycache__/mod.py",
            ".venv/lib.py",
            "node_modules/pkg/index.js",
            ".env",
            ".env.local",
            "certs/private.key",
        ]
        for rel in ignored_files:
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("NEEDLE_SECRET\n", encoding="utf-8")

        (self.root / "binary.dat").write_bytes(b"NEEDLE_SECRET\x00binary_payload")
        (self.root / "visible.txt").write_text("NEEDLE_SECRET visible\n", encoding="utf-8")

        out = tool_grep("NEEDLE_SECRET", path=str(self.root))
        self.assertEqual(out, "visible.txt:1: NEEDLE_SECRET visible")

    def test_max_results_cap_and_zero_matches(self) -> None:
        target = self.root / "many.txt"
        target.write_text("\n".join(f"match line {i}" for i in range(1, 11)) + "\n", encoding="utf-8")

        capped = tool_grep("match line", path=str(self.root), max_results=3)
        self.assertEqual(
            capped.splitlines(),
            [
                "many.txt:1: match line 1",
                "many.txt:2: match line 2",
                "many.txt:3: match line 3",
                "[limite de 3 resultados atingido]",
            ],
        )

        exact = tool_grep("match line", path=str(self.root), max_results=10)
        self.assertEqual(len(exact.splitlines()), 10)
        self.assertNotIn("[limite de", exact)

        none_out = tool_grep("DOES_NOT_EXIST_XYZ", path=str(self.root))
        self.assertEqual(none_out, "Nenhuma ocorrência encontrada.")

    def test_error_handling_for_empty_pattern_invalid_limit_and_missing_path(self) -> None:
        self.assertEqual(
            tool_grep("", path=str(self.root)),
            "ERRO: padrão de busca não pode ser vazio",
        )
        self.assertEqual(
            tool_grep("abc", path=str(self.root), max_results=0),
            "ERRO: max_results deve ser maior que zero",
        )
        self.assertEqual(
            tool_grep("abc", path=str(self.root), max_results=-1),
            "ERRO: max_results deve ser maior que zero",
        )
        missing = self.root / "no_such_dir"
        err = tool_grep("abc", path=str(missing))
        self.assertTrue(err.startswith("ERRO: caminho não encontrado:"))

    def test_performance_200_files_under_500ms(self) -> None:
        for d in range(10):
            subdir = self.root / f"dir_{d:02d}"
            subdir.mkdir()
            for f in range(20):
                content = f"line 1\nline 2\nfile {d}_{f}\n"
                if d == 9 and f == 19:
                    content += "NEEDLE_PERF_TARGET\n"
                (subdir / f"file_{f:02d}.py").write_text(content, encoding="utf-8")

        tool_grep("__WARMUP_ABSENT_TOKEN__", path=str(self.root))

        start = time.perf_counter()
        out = tool_grep("NEEDLE_PERF_TARGET", path=str(self.root))
        elapsed = time.perf_counter() - start

        self.assertEqual(out, "dir_09/file_19.py:4: NEEDLE_PERF_TARGET")
        self.assertLess(elapsed, 0.5, f"grep over 200 files took {elapsed:.3f}s (>= 0.5s)")


class TestToolListDir(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self._tmp.name).resolve()
        self.root = base / "workspace"
        self.outside = base / "outside"
        self.root.mkdir()
        self.outside.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_depth_1_vs_depth_2_and_3_with_trailing_slash_on_dirs(self) -> None:
        (self.root / "alpha.py").write_text("a = 1\n", encoding="utf-8")
        (self.root / "z.txt").write_text("z\n", encoding="utf-8")
        deep = self.root / "pkg" / "sub"
        deep.mkdir(parents=True)
        (self.root / "pkg" / "beta.py").write_text("b = 2\n", encoding="utf-8")
        (deep / "gamma.py").write_text("c = 3\n", encoding="utf-8")

        depth1 = tool_list_dir(path=str(self.root), max_depth=1)
        self.assertEqual(depth1.splitlines(), ["alpha.py", "pkg/", "z.txt"])

        depth2 = tool_list_dir(path=str(self.root), max_depth=2)
        self.assertEqual(
            depth2.splitlines(),
            ["alpha.py", "pkg/", "  beta.py", "  sub/", "z.txt"],
        )

        depth3 = tool_list_dir(path=str(self.root), max_depth=3)
        self.assertEqual(
            depth3.splitlines(),
            ["alpha.py", "pkg/", "  beta.py", "  sub/", "    gamma.py", "z.txt"],
        )

    def test_skips_ignored_dirs_sensitive_files_and_escaping_links(self) -> None:
        for rel in (
            ".git/config",
            "__pycache__/mod.pyc",
            ".venv/lib.py",
            "node_modules/pkg/index.js",
            ".env",
            ".env.local",
            "certs/private.key",
        ):
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("secret\n", encoding="utf-8")

        (self.root / "visible.py").write_text("ok = True\n", encoding="utf-8")
        outside_file = self.outside / "leaked.txt"
        outside_file.write_text("outside\n", encoding="utf-8")
        link_created = _create_escaping_link(
            self.root, self.outside, outside_file, "escape_link.txt", "escape_dir"
        )

        out = tool_list_dir(path=str(self.root), max_depth=3)
        self.assertEqual(out.splitlines(), ["certs/", "visible.py"])
        self.assertNotIn(".env", out)
        self.assertNotIn(".git", out)
        self.assertNotIn("__pycache__", out)
        self.assertNotIn("node_modules", out)
        self.assertNotIn("private.key", out)
        if link_created is not None:
            self.assertNotIn(link_created, out)
            self.assertNotIn("leaked.txt", out)

    def test_empty_directory_and_only_hidden_ignored_entries(self) -> None:
        empty_dir = self.root / "empty"
        empty_dir.mkdir()
        self.assertEqual(tool_list_dir(path=str(empty_dir)), "Diretório vazio.")

        (empty_dir / ".env").write_text("KEY=1\n", encoding="utf-8")
        (empty_dir / "__pycache__").mkdir()
        self.assertEqual(tool_list_dir(path=str(empty_dir)), "Diretório vazio.")

    def test_error_handling_for_invalid_depth_missing_path_and_file_path(self) -> None:
        self.assertEqual(
            tool_list_dir(path=str(self.root), max_depth=0),
            "ERRO: max_depth deve ser maior que zero",
        )
        self.assertEqual(
            tool_list_dir(path=str(self.root), max_depth=-2),
            "ERRO: max_depth deve ser maior que zero",
        )

        missing = self.root / "does_not_exist"
        self.assertEqual(
            tool_list_dir(path=str(missing)),
            f"ERRO: caminho não encontrado: {missing}",
        )

        file_path = self.root / "single.txt"
        file_path.write_text("hi\n", encoding="utf-8")
        self.assertEqual(
            tool_list_dir(path=str(file_path)),
            f"ERRO: o caminho não é um diretório: {file_path}",
        )


class TestToolFindFiles(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self._tmp.name).resolve()
        self.root = base / "workspace"
        self.outside = base / "outside"
        self.root.mkdir()
        self.outside.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_filename_and_relative_path_glob_patterns(self) -> None:
        for rel in (
            "app.py",
            "docker-compose.yml",
            "deploy/docker-compose.prod.yml",
            "pkg/mod.py",
            "pkg/sub/helper.py",
            "pkg/readme.md",
            "other/script.py",
        ):
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\n", encoding="utf-8")

        py_files = tool_find_files(pattern="*.py", path=str(self.root))
        self.assertEqual(
            py_files.splitlines(),
            ["app.py", "other/script.py", "pkg/mod.py", "pkg/sub/helper.py"],
        )

        compose_files = tool_find_files(pattern="docker-compose*.yml", path=str(self.root))
        self.assertEqual(
            compose_files.splitlines(),
            ["deploy/docker-compose.prod.yml", "docker-compose.yml"],
        )

        pkg_py = tool_find_files(pattern="pkg/*.py", path=str(self.root))
        self.assertEqual(pkg_py.splitlines(), ["pkg/mod.py", "pkg/sub/helper.py"])

        pkg_py_win = tool_find_files(pattern=r"pkg\*.py", path=str(self.root))
        self.assertEqual(pkg_py_win.splitlines(), ["pkg/mod.py", "pkg/sub/helper.py"])

    def test_skips_ignored_dirs_sensitive_files_and_escaping_links(self) -> None:
        for rel in (
            ".git/config",
            "__pycache__/mod.py",
            ".venv/lib.py",
            "node_modules/pkg/index.js",
            ".env",
            ".env.local",
            "certs/private.key",
            "src/app.py",
        ):
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("content\n", encoding="utf-8")

        outside_file = self.outside / "leaked.py"
        outside_file.write_text("outside\n", encoding="utf-8")
        link_created = _create_escaping_link(
            self.root, self.outside, outside_file, "escape_link.py", "escape_dir"
        )

        out = tool_find_files(pattern="*", path=str(self.root))
        self.assertEqual(out, "src/app.py")
        if link_created is not None:
            self.assertNotIn("leaked.py", out)
            self.assertNotIn(link_created, out)

    def test_zero_matches_and_error_handling(self) -> None:
        (self.root / "notes.txt").write_text("hello\n", encoding="utf-8")

        self.assertEqual(
            tool_find_files(pattern="*.rs", path=str(self.root)),
            "Nenhum arquivo encontrado.",
        )
        self.assertEqual(
            tool_find_files(pattern="", path=str(self.root)),
            "ERRO: padrão de busca não pode ser vazio",
        )
        self.assertEqual(
            tool_find_files(pattern="   ", path=str(self.root)),
            "ERRO: padrão de busca não pode ser vazio",
        )

        missing = self.root / "missing_dir"
        self.assertEqual(
            tool_find_files(pattern="*", path=str(missing)),
            f"ERRO: caminho não encontrado: {missing}",
        )

        file_path = self.root / "notes.txt"
        self.assertEqual(
            tool_find_files(pattern="*", path=str(file_path)),
            f"ERRO: o caminho não é um diretório: {file_path}",
        )

    def test_performance_500_files_under_300ms(self) -> None:
        for d in range(25):
            subdir = self.root / f"dir_{d:02d}"
            subdir.mkdir()
            for f in range(20):
                ext = ".py" if f == 19 else ".txt"
                (subdir / f"file_{f:02d}{ext}").write_text("x\n", encoding="utf-8")

        tool_find_files(pattern="*.__warmup__", path=str(self.root))

        start = time.perf_counter()
        out = tool_find_files(pattern="*.py", path=str(self.root))
        elapsed = time.perf_counter() - start

        lines = out.splitlines()
        self.assertEqual(len(lines), 25)
        self.assertEqual(lines[0], "dir_00/file_19.py")
        self.assertEqual(lines[-1], "dir_24/file_19.py")
        self.assertLess(elapsed, 0.3, f"find_files over 500 files took {elapsed:.3f}s (>= 0.3s)")


class TestToolBash(unittest.TestCase):
    def test_captures_exit_zero_stdout_and_stderr(self) -> None:
        cmd = f'"{sys.executable}" -c "import sys; print(\'out_msg\'); print(\'err_msg\', file=sys.stderr)"'
        out = tool_bash(cmd)
        self.assertTrue(out.startswith("exit=0\n"))
        self.assertIn("stdout:\nout_msg", out)
        self.assertIn("stderr:\nerr_msg", out)

    def test_captures_nonzero_exit_code(self) -> None:
        cmd = f'"{sys.executable}" -c "import sys; sys.exit(7)"'
        out = tool_bash(cmd)
        self.assertTrue(out.startswith("exit=7\n"))

    def test_tool_bash_async_does_not_block_concurrent_event_loop_tasks(self) -> None:
        self.assertIs(TOOLS["bash"], tool_bash_async)
        cmd = f'"{sys.executable}" -c "import time; time.sleep(0.25); print(\'async_done\')"'

        async def run_concurrently() -> tuple[str, int]:
            done = asyncio.Event()
            ticks = 0

            async def ticker() -> None:
                nonlocal ticks
                while not done.is_set():
                    ticks += 1
                    await asyncio.sleep(0.02)

            async def run_shell() -> str:
                try:
                    return await tool_bash_async(cmd)
                finally:
                    done.set()

            out, _ = await asyncio.gather(run_shell(), ticker())
            return out, ticks

        out, ticks = asyncio.run(run_concurrently())
        self.assertTrue(out.startswith("exit=0\n"))
        self.assertIn("async_done", out)
        self.assertGreaterEqual(ticks, 3)


class TestDiscoverContext(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_collects_agents_md_from_parent_to_child_in_order(self) -> None:
        parent_dir = self.root / "workspace"
        mid_dir = parent_dir / "pkg"
        leaf_dir = mid_dir / "sub"
        leaf_dir.mkdir(parents=True)

        parent_md = parent_dir / "AGENTS.md"
        leaf_md = leaf_dir / "AGENTS.md"
        parent_md.write_text("ROOT_RULES", encoding="utf-8")
        leaf_md.write_text("LEAF_RULES", encoding="utf-8")

        ctx = discover_context(leaf_dir)

        parent_tag = f"<ctx path='{parent_md}'>\nROOT_RULES\n</ctx>"
        leaf_tag = f"<ctx path='{leaf_md}'>\nLEAF_RULES\n</ctx>"
        self.assertIn(parent_tag, ctx)
        self.assertIn(leaf_tag, ctx)
        self.assertLess(ctx.index(parent_tag), ctx.index(leaf_tag))
        self.assertNotIn(str(mid_dir / "AGENTS.md"), ctx)


class TestToolSpecsApprovalMetadata(unittest.TestCase):
    def test_approval_required_tools_and_clean_schemas(self) -> None:
        self.assertEqual(APPROVAL_REQUIRED_TOOLS, frozenset({"bash", "write_file", "search_replace"}))
        for schema in TOOL_SCHEMAS:
            self.assertEqual(set(schema.keys()), {"name", "description", "parameters"})


class TestWorkspaceBoundTools(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self._tmp.name).resolve()
        self.root_a = base / "ws_a"
        self.root_b = base / "ws_b"
        self.root_a.mkdir()
        self.root_b.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _extract_bash_stdout(output: str) -> str:
        after_stdout = output.split("stdout:\n", 1)[1]
        return after_stdout.split("\nstderr:\n", 1)[0].strip()

    def test_file_and_search_tools_resolve_relative_paths_against_root_dir_without_chdir(self) -> None:
        cwd_before = pathlib.Path.cwd()

        tool_write_file("sub/item.txt", "alpha_1\nalpha_2\n", root_dir=self.root_a)
        tool_write_file("sub/item.txt", "beta_1\n", root_dir=self.root_b)
        tool_write_file("only_a.txt", "a\n", root_dir=self.root_a)
        tool_write_file("only_b.txt", "b\n", root_dir=self.root_b)

        self.assertEqual(
            (self.root_a / "sub" / "item.txt").read_text(encoding="utf-8"),
            "alpha_1\nalpha_2\n",
        )
        self.assertEqual(
            (self.root_b / "sub" / "item.txt").read_text(encoding="utf-8"),
            "beta_1\n",
        )

        read_a = tool_read_file("sub/item.txt", root_dir=self.root_a)
        read_b = tool_read_file("sub/item.txt", root_dir=self.root_b)
        self.assertIn("alpha_1", read_a)
        self.assertNotIn("beta_1", read_a)
        self.assertIn("beta_1", read_b)
        self.assertNotIn("alpha_1", read_b)

        self.assertEqual(
            tool_search_replace("sub/item.txt", "alpha_2", "alpha_updated", root_dir=self.root_a),
            "OK",
        )
        self.assertEqual(
            (self.root_b / "sub" / "item.txt").read_text(encoding="utf-8"),
            "beta_1\n",
        )

        self.assertEqual(
            tool_grep("alpha_updated", path=".", root_dir=self.root_a),
            "sub/item.txt:2: alpha_updated",
        )
        self.assertEqual(
            tool_grep("alpha_updated", path=".", root_dir=self.root_b),
            "Nenhuma ocorrência encontrada.",
        )
        self.assertEqual(
            tool_grep("beta_1", path="sub/item.txt", root_dir=self.root_b),
            "sub/item.txt:1: beta_1",
        )

        list_a = tool_list_dir(".", max_depth=2, root_dir=self.root_a)
        list_b = tool_list_dir(".", max_depth=2, root_dir=self.root_b)
        self.assertIn("only_a.txt", list_a)
        self.assertNotIn("only_b.txt", list_a)
        self.assertIn("only_b.txt", list_b)
        self.assertNotIn("only_a.txt", list_b)

        self.assertEqual(
            tool_find_files("*.txt", path=".", root_dir=self.root_a).splitlines(),
            ["only_a.txt", "sub/item.txt"],
        )
        self.assertEqual(
            tool_find_files("*.txt", path=".", root_dir=self.root_b).splitlines(),
            ["only_b.txt", "sub/item.txt"],
        )
        self.assertEqual(pathlib.Path.cwd(), cwd_before)

    def test_bash_and_bash_async_respect_explicit_cwd_without_chdir(self) -> None:
        cwd_before = pathlib.Path.cwd()
        cmd = f'"{sys.executable}" -c "import os; print(os.getcwd())"'

        sync_out = tool_bash(cmd, cwd=self.root_a)
        async_out = asyncio.run(tool_bash_async(cmd, cwd=self.root_b))

        self.assertTrue(sync_out.startswith("exit=0\n"))
        self.assertTrue(async_out.startswith("exit=0\n"))
        self.assertEqual(
            pathlib.Path(self._extract_bash_stdout(sync_out)).resolve(),
            self.root_a,
        )
        self.assertEqual(
            pathlib.Path(self._extract_bash_stdout(async_out)).resolve(),
            self.root_b,
        )
        self.assertEqual(pathlib.Path.cwd(), cwd_before)

    def test_build_tools_binds_all_native_tools_and_blocks_kwarg_injection(self) -> None:
        tools_a = build_tools(self.root_a)
        tools_b = build_tools(self.root_b)
        tools_default = build_tools(None)

        self.assertEqual(set(tools_a.keys()), set(TOOLS.keys()))
        self.assertEqual(set(tools_b.keys()), set(TOOLS.keys()))
        self.assertEqual(set(tools_default.keys()), set(TOOLS.keys()))
        for schema in TOOL_SCHEMAS:
            props = schema["parameters"]["properties"]
            self.assertNotIn("root_dir", props)
            self.assertNotIn("cwd", props)

        tools_a["write_file"](path="note.txt", content="from_a_v1\n")
        tools_b["write_file"](path="note.txt", content="from_b_v1\n")
        self.assertIn("from_a_v1", tools_a["read_file"](path="note.txt"))
        self.assertIn("from_b_v1", tools_b["read_file"](path="note.txt"))

        self.assertEqual(
            tools_a["search_replace"](path="note.txt", search="from_a_v1", replace="from_a_v2"),
            "OK",
        )
        self.assertEqual(tools_a["grep"](pattern="from_a_v2", path="."), "note.txt:1: from_a_v2")
        self.assertEqual(tools_a["list_dir"](path="."), "note.txt")
        self.assertEqual(tools_a["find_files"](pattern="*.txt", path="."), "note.txt")

        cmd = f'"{sys.executable}" -c "import os; print(os.getcwd())"'

        async def run_bash_pair() -> tuple[str, str]:
            return await asyncio.gather(tools_a["bash"](cmd), tools_b["bash"](cmd))

        bash_a, bash_b = asyncio.run(run_bash_pair())
        self.assertEqual(pathlib.Path(self._extract_bash_stdout(bash_a)).resolve(), self.root_a)
        self.assertEqual(pathlib.Path(self._extract_bash_stdout(bash_b)).resolve(), self.root_b)

        with self.assertRaises(TypeError):
            tools_a["read_file"](path="note.txt", root_dir=str(self.root_b))
        with self.assertRaises(TypeError):
            tools_default["read_file"](path="note.txt", root_dir=str(self.root_b))
        with self.assertRaises(TypeError):
            asyncio.run(tools_a["bash"](cmd="echo ok", cwd=str(self.root_b)))
        with self.assertRaises(TypeError):
            asyncio.run(tools_default["bash"](cmd="echo ok", cwd=str(self.root_b)))


if __name__ == "__main__":
    unittest.main()
