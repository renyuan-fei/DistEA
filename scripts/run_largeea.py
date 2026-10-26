"""
run_largeea.py -- launcher for LargeEA's src/main.py that redirects its
hardcoded DBP1M ("large" dataset) data path and patches torch/transformers
incompatibilities.

Reads:  DBP1M in LargeScaleEAData's layout at MKDATA_DIR.
Writes: whatever LargeEA's main.py writes ('tmp4'/'unsup' folders, logs).

Same issue and same fix as run_clusterea.py: `LargeScaleEAData('../mkdata/',
...)` is hardcoded in main.py itself (not exposed via CLI), and that path is
inside LargeEA's own repo, which this launcher leaves untouched.

LargeEA's own dataset.py::LargeScaleEAData is structurally identical to
ClusterEA's class of the same name (same {lang}_triples{1,2}/{lang}_
ent_links file-naming convention) -- reuses DistEA's data/clusterea_mkdata/
directory directly, no separate data prep.

Usage (from inside LargeEA/src/, matching main.py's own relative
expectations for small/medium scale datasets and its 'tmp4'/'unsup' output
folders):
    python <DistEA>/scripts/run_largeea.py main.py --phase 0 --dataset large --lang de --model rrea
(all args after the script name are passed through unchanged)
"""

import os
import runpy
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DBP1M in LargeEA's mkdata layout; override with the DISTEA_MKDATA_DIR environment variable.
MKDATA_DIR = os.environ.get("DISTEA_MKDATA_DIR", os.path.join(_REPO, "data", "clusterea_mkdata")) + "/"


def install_torch_six_shim() -> None:
    """
    torch_geometric==1.6.1 (LargeEA's own pin) does
    `from torch._six import container_abcs, string_classes, int_classes` in
    several internal modules (data_parallel.py, dataloader.py, ...). These
    were real Python-2/3-compat aliases in old torch (container_abcs ==
    collections.abc, string_classes == (str,), int_classes == (int,)) that
    later torch versions dropped from _six entirely (ImportError with
    torch 1.10.2, used because LargeEA's own pin of torch==1.7.0 predates
    broad CUDA 11 support). Restore the three names torch_geometric 1.6.1
    actually imports, rather than downgrading torch (which would reopen
    the CUDA 11 wheel-availability problem).
    """
    import collections.abc

    import torch._six as six_module

    six_module.container_abcs = collections.abc
    six_module.string_classes = (str,)
    six_module.int_classes = (int,)


def install_lr_scheduler_shim() -> None:
    """
    Upstream incompatibility: main.py imports
    text_sim -> emb_loader -> `from transformers import *`, whose own
    trainer_pt_utils.py does
    `from torch.optim.lr_scheduler import SAVE_STATE_WARNING` --
    SAVE_STATE_WARNING was just a deprecation-warning message string in
    older torch, dropped entirely in torch 1.10.2. Restoring it
    as a placeholder string is behavior-preserving (transformers only ever
    used it as literal warning text, never as a functional flag) --
    same class of fix as run_clusterea.py's np.int shim.
    """
    import torch.optim.lr_scheduler as lr_scheduler_module

    if not hasattr(lr_scheduler_module, "SAVE_STATE_WARNING"):
        lr_scheduler_module.SAVE_STATE_WARNING = (
            "Please also save or load the state of the optimizer when saving or loading the scheduler."
        )


def install_mkdata_path_patch() -> None:
    import dataset as dataset_module

    real_large_scale_init = dataset_module.LargeScaleEAData.__init__

    def patched_init(self, path, lang, strict=False, shuffle_pairs=False, train_ratio=0.2, **kwargs):
        return real_large_scale_init(self, MKDATA_DIR, lang, strict, shuffle_pairs, train_ratio, **kwargs)

    dataset_module.LargeScaleEAData.__init__ = patched_init


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_largeea.py <target_script.py> [args...]")

    target_script = sys.argv[1]
    sys.argv = sys.argv[1:]

    sys.path.insert(0, os.getcwd())

    install_torch_six_shim()
    install_lr_scheduler_shim()
    install_mkdata_path_patch()
    runpy.run_path(target_script, run_name="__main__")


if __name__ == "__main__":
    main()
