import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--quick", action="store_true", help="small/fast smoke-test settings")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--seeds", type=int, default=None)
    return ap


def save(out, name, obj, fig=None):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(obj, f, indent=2)
    if fig is not None:
        fig.savefig(os.path.join(out, name + ".png"), dpi=130, bbox_inches="tight")
    print(f"saved {name} -> {os.path.abspath(out)}")
