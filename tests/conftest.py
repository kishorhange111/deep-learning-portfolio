"""Shared pytest setup.

* Makes the repository root importable (`import common`).
* Runs Keras on the PyTorch backend so the Keras projects can be tested on a
  CPU-only machine without installing TensorFlow.
* `load_train(...)` imports a project's train.py as a module (its `main()` is
  guarded by `if __name__ == "__main__"`, so nothing is trained or downloaded)
  and skips the test when an optional heavy dependency is not installed.
  `load_module(...)` does the same for any other file in a project folder.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(relpath, filename):
    """Import <ROOT>/<relpath>/<filename> under a unique module name, or skip if a third-party
    dependency is missing. Modules import their sibling files by plain name, so those must be
    unique across the repository."""
    path = ROOT / relpath / filename
    name = pathlib.Path(filename).stem + "_" + relpath.replace("/", "_").replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] == "common":
            raise
        pytest.skip(f"optional dependency not installed: {e.name}", allow_module_level=True)
    sys.modules[name] = module
    return module


def load_train(relpath):
    """Import <ROOT>/<relpath>/train.py as a module (see load_module)."""
    return load_module(relpath, "train.py")
