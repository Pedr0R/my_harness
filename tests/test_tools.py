from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

from agent import (
    _truncate,
    discover_context,
    tool_bash,
    tool_read_file,
    tool_search_replace,
    tool_write_file,
)


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


if __name__ == "__main__":
    unittest.main()
