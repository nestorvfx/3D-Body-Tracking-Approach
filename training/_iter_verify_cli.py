import sys
sys.path.insert(0, "training")
import argparse
sys.argv = ["train.py", "--dataset-dir", "x", "--out-dir", "y",
            "--occluder-dir", "a", "--bg-corpus-dir", "b",
            "--matte-dir", "c", "--fda-refs-dir", "d"]
from train import parse_args   # noqa: E402
ns = parse_args()
print("OK:", ns.occluder_dir, ns.bg_corpus_dir, ns.matte_dir, ns.fda_refs_dir,
      ns.p_occluder, ns.p_bg_composite, ns.p_fda)
