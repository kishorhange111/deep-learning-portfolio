"""
A byte-level text corpus that every machine already has: the Python standard library's own source.

The optimizer, mixture-of-experts and quantization projects train small byte-level language
models on it, so they need no downloads. Files are split into train and validation by a hash of
their relative path, so the split is stable and no file is in both. The corpus depends on the
installed Python version, so `corpus_info` records the version, the file counts and a SHA-256
fingerprint with every result.
"""

import hashlib
import os
import sys
import sysconfig

import numpy as np

SKIP_DIRS = {"site-packages", "dist-packages", "__pycache__", "idlelib", "turtledemo", "pydoc_data"}


def stdlib_files(root=None):
    """Sorted relative paths of every .py file under the standard library (third-party packages excluded)."""
    root = root or sysconfig.get_paths()["stdlib"]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(".py"):
                out.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(out)


def is_validation(relpath, val_percent=5):
    """Stable file-level split: about `val_percent` % of files go to validation."""
    return int(hashlib.sha256(relpath.encode()).hexdigest(), 16) % 100 < val_percent


def load_corpus(root=None, val_percent=5, max_bytes=None):
    """(train, val, info): uint8 arrays of concatenated file contents (files separated by a NUL byte)."""
    root = root or sysconfig.get_paths()["stdlib"]
    parts = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    digest = hashlib.sha256()
    for rel in stdlib_files(root):
        with open(os.path.join(root, rel), "rb") as f:
            data = f.read()
        split = "val" if is_validation(rel, val_percent) else "train"
        parts[split].append(data + b"\0")
        counts[split] += 1
        digest.update(rel.encode() + b"\0" + data)
    train = np.frombuffer(b"".join(parts["train"]), dtype=np.uint8)
    val = np.frombuffer(b"".join(parts["val"]), dtype=np.uint8)
    if max_bytes:
        train = train[:max_bytes]
    info = {
        "source": "Python standard library .py files",
        "python": sys.version.split()[0],
        "root": root,
        "train_files": counts["train"],
        "val_files": counts["val"],
        "train_bytes": int(len(train)),
        "val_bytes": int(len(val)),
        "sha256": digest.hexdigest(),
    }
    return train, val, info


def synthetic_corpus(n_bytes=200_000, seed=0):
    """A small repetitive byte corpus with Python-like structure, for tests (no file system access)."""
    rng = np.random.default_rng(seed)
    words = [b"def ", b"return ", b"self", b"import ", b"class ", b"for ", b"in ", b"if ", b"else:", b"None"]
    names = [b"x", b"y", b"value", b"items", b"key", b"data", b"index"]
    out = bytearray()
    while len(out) < n_bytes:
        indent = b"    " * int(rng.integers(0, 3))
        line = indent + words[rng.integers(len(words))] + names[rng.integers(len(names))]
        line += b"(" + names[rng.integers(len(names))] + b", " + str(int(rng.integers(100))).encode() + b")\n"
        out += line
    data = np.frombuffer(bytes(out[:n_bytes]), dtype=np.uint8)
    cut = int(0.9 * n_bytes)
    return data[:cut], data[cut:], {"source": "synthetic", "train_bytes": cut, "val_bytes": n_bytes - cut}


def random_windows(data, batch, seq_len, rng):
    """(x, y) int64 arrays of shape (batch, seq_len): random windows and the same shifted by one byte."""
    idx = rng.integers(0, len(data) - seq_len - 1, size=batch)
    w = np.stack([data[i : i + seq_len + 1] for i in idx]).astype(np.int64)
    return w[:, :-1], w[:, 1:]


def fixed_windows(data, n, seq_len, seed=0):
    """A fixed set of n evaluation windows (the same every call with the same seed)."""
    return random_windows(data, n, seq_len, np.random.default_rng(seed))


BYTE_CLASSES = (
    "lowercase",
    "uppercase",
    "digit",
    "space",
    "newline",
    "underscore",
    "quote",
    "bracket",
    "operator",
    "other",
)


def byte_class(b):
    """Coarse class of a byte, for routing and error analyses (bytes >= 128 are 'other': UTF-8 pieces)."""
    if b >= 128:
        return "other"
    c = chr(b)
    if c.islower():
        return "lowercase"
    if c.isupper():
        return "uppercase"
    if c.isdigit():
        return "digit"
    if c == "\n":
        return "newline"
    if c in " \t":
        return "space"
    if c == "_":
        return "underscore"
    if c in "'\"":
        return "quote"
    if c in "()[]{}":
        return "bracket"
    if c in "+-*/%=<>!&|^~.,:;@#":
        return "operator"
    return "other"


BYTE_CLASS_TABLE = np.array([BYTE_CLASSES.index(byte_class(b)) for b in range(256)], dtype=np.int64)
