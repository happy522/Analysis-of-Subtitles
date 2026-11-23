"""One-time setup for compute_ted_astred.py (creates TED_ASTrED/.deps/, which is git-ignored).

1. Installs awesome-align 0.1.7 into .deps/pylib (no dependencies; torch is already required by the pipeline).
2. Builds .deps/mbert: bert-base-multilingual-cased in the legacy format awesome-align can load
   (pytorch_model.bin instead of model.safetensors) + config.json, vocab.txt, tokenizer_config.json.
   The weights are the unchanged mBERT weights, only the file format differs.

Run:  python TED_ASTrED/setup_awesome_align.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

DEPS = Path(__file__).resolve().parent / ".deps"
PYLIB = DEPS / "pylib"
MBERT = DEPS / "mbert"
MODEL = "bert-base-multilingual-cased"


def install_awesome_align():
    if (PYLIB / "awesome_align").exists():
        print("awesome-align already installed in", PYLIB)
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(PYLIB),
                           "awesome-align==0.1.7"])


def build_mbert():
    if (MBERT / "pytorch_model.bin").exists():
        print("mBERT already prepared in", MBERT)
        return
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    snap = Path(snapshot_download(MODEL, allow_patterns=["config.json", "vocab.txt", "tokenizer_config.json",
                                                         "model.safetensors"]))
    MBERT.mkdir(parents=True, exist_ok=True)
    for f in ["config.json", "vocab.txt", "tokenizer_config.json"]:
        shutil.copy(snap / f, MBERT / f)
    torch.save(load_file(str(snap / "model.safetensors")), MBERT / "pytorch_model.bin")
    print("mBERT prepared in", MBERT)


if __name__ == "__main__":
    install_awesome_align()
    build_mbert()
