import ast
import unittest
from pathlib import Path


class PublicDocstringTests(unittest.TestCase):
    def test_public_modules_classes_functions_and_methods_have_docstrings(self) -> None:
        missing: list[tuple[str, str, str]] = []
        for path in sorted(Path("colophon").glob("*.py")):
            module = ast.parse(path.read_text(encoding="utf-8"))
            if ast.get_docstring(module) is None:
                missing.append((str(path), "module", path.stem))

            for node in module.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_"):
                        continue
                    if ast.get_docstring(node) is None:
                        missing.append((str(path), "function", node.name))
                    continue

                if isinstance(node, ast.ClassDef):
                    if node.name.startswith("_"):
                        continue
                    if ast.get_docstring(node) is None:
                        missing.append((str(path), "class", node.name))

                    for member in node.body:
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if member.name.startswith("_"):
                                continue
                            if ast.get_docstring(member) is None:
                                missing.append((str(path), f"method:{node.name}", member.name))

        if missing:
            formatted = "\n".join(f"{path}\t{kind}\t{name}" for path, kind, name in missing)
            self.fail(f"Missing public docstrings:\n{formatted}")


if __name__ == "__main__":
    unittest.main()
