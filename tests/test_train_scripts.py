"""Structural checks for every training script (no GPU, no downloads).

* valid Python with a module docstring;
* every function and class has a docstring;
* a `__main__` guard, so importing a script never starts a training run;
* scripts whose dependencies are installed import cleanly (their `common.*` imports resolve).
"""

import ast

import pytest
from conftest import ROOT, load_train

SCRIPTS = sorted(ROOT.glob("*/*/train.py"))


def test_all_projects_found():
    assert len(SCRIPTS) == 14


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.parent.name)
def test_docstrings_and_main_guard(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), "module docstring explaining the project is required"
    assert any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("__"):
            assert ast.get_docstring(node), f"{node.name} (line {node.lineno}) needs a docstring"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.parent.name)
def test_imports_without_running(path):
    module = load_train(str(path.parent.relative_to(ROOT)))
    assert hasattr(module, "main") or hasattr(module, "compare")
