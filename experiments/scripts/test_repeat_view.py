#!/usr/bin/env python3
"""
Regression test for the repeat-frame consistency window (`duplicate_first_view`).

The claim being tested is that duplicating a view needs no new loss code: the
duplicate's ground truth is identical, so the existing regression loss already
punishes the model for predicting something different the second time.  That is
only true if (a) the duplicate really reaches the model as an extra view, (b) the
loss is finite and larger than zero on it, and (c) gradients flow.  If any of
those silently fails, a training run would just be 25% slower for nothing.

    python tools/ab/test_repeat_view.py --ckpt <training checkpoint>

Runs on CPU-sized synthetic views unless --ckpt is given.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def structural_tests():
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r("")
    from dust3r.inference import duplicate_view_in_batch

    print("\n[1] duplicate_view_in_batch")
    t = torch.randn(2, 3)
    batch = [{"img": t, "tag": i} for i in range(4)]
    out = duplicate_view_in_batch(batch, src=0, at=1)
    check("window grew by one", len(out) == 5, f"{len(batch)} -> {len(out)}")
    check("the copy sits at the requested position", [v["tag"] for v in out] == [0, 0, 1, 2, 3],
          str([v["tag"] for v in out]))
    check("the copy shares its tensors", out[0]["img"] is out[1]["img"])
    check("the copy is a distinct dict", out[0] is not out[1])
    out[1]["written_by_model"] = True
    check("a key written into the copy does not leak into the original",
          "written_by_model" not in out[0])
    check("the input list is not mutated", len(batch) == 4)

    check("src/at are clamped, not asserted",
          len(duplicate_view_in_batch(batch, src=99, at=99)) == 5)
    check("an empty batch is a no-op", duplicate_view_in_batch([]) == [])

    print("\n[2] the trainer reads the flag")
    import inspect
    import train as T

    src = inspect.getsource(T.train_one_epoch)
    check("duplicate_first_view is read from args", 'getattr(args, "duplicate_first_view"' in src)
    check("the insert happens before the loss dispatch",
          src.index("duplicate_view_in_batch(batch") < src.index("result = loss_of_one_batch"))
    check("validation is untouched",
          "duplicate_view_in_batch" not in inspect.getsource(T.test_one_epoch))


def loss_test(ckpt, model_cfg, device, seq, nframes, size):
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(ckpt)
    import demo as D
    from src.dust3r.inference import (
        duplicate_view_in_batch,
        loss_of_one_batch,
        _move_batch_to_model,
    )
    from src.dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig
    from src.dust3r.utils.safe_eval import safe_eval_expr

    print(f"\n[3] loss and gradients on a duplicated window")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_str = model_cfg or getattr(sd.get("args", None), "model", None)
    crit_str = getattr(sd.get("args", None), "train_criterion", None)
    if not cfg_str:
        raise SystemExit("no model config embedded in the checkpoint; pass --model-cfg")
    print(f"       criterion: {crit_str}")

    net = safe_eval_expr(cfg_str, {"ARCroco3DStereo": ARCroco3DStereo,
                                   "ARCroco3DStereoConfig": ARCroco3DStereoConfig,
                                   "inf": float("inf")})
    rep = net.load_state_dict(sd.get("model", sd), strict=False)
    check("checkpoint loads cleanly", not rep.missing_keys and not rep.unexpected_keys,
          f"{len(rep.missing_keys)} missing / {len(rep.unexpected_keys)} unexpected")
    net = net.to(device).train()

    import src.dust3r.losses as L

    ns = {k: getattr(L, k) for k in dir(L) if not k.startswith("_")}
    ns["inf"] = float("inf")
    crit = safe_eval_expr(crit_str, ns)
    crit = crit.to(device)

    frames = sorted(os.path.join(seq, f) for f in os.listdir(seq)
                    if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))[:nframes]
    if len(frames) < 2:
        raise SystemExit(f"need >= 2 frames in {seq}")

    def make_batch():
        views = _move_batch_to_model(D.prepare_input(frames, [True] * len(frames), size), net)
        # the regression loss needs ground truth; synthesise it from the model's
        # own prediction so the test exercises the loss path without a dataset
        with torch.no_grad():
            res = net(views)
        out = []
        for v, p in zip(res.views, res.ress):
            v = dict(v)
            v["pts3d"] = p["pts3d_in_other_view"].detach().clone()
            b = v["pts3d"].shape[0]
            v["valid_mask"] = torch.ones(v["pts3d"].shape[:-1], dtype=torch.bool,
                                         device=v["pts3d"].device)
            # the remaining fields the regression loss reads off the ground
            # truth; values are arbitrary but must be present and consistent
            v["camera_only"] = torch.zeros(b, dtype=torch.bool, device=device)
            v["depth_only"] = torch.zeros(b, dtype=torch.bool, device=device)
            v["single_view"] = torch.zeros(b, dtype=torch.bool, device=device)
            v["reset"] = torch.zeros(b, dtype=torch.bool, device=device)
            v["is_metric"] = torch.ones(b, dtype=torch.bool, device=device)
            v["is_metric_scale"] = torch.ones(b, dtype=torch.bool, device=device)
            v["img_mask"] = torch.ones(b, dtype=torch.bool, device=device)
            v["ray_mask"] = torch.zeros(b, dtype=torch.bool, device=device)
            v["sky_mask"] = torch.zeros_like(v["valid_mask"])
            out.append(v)
        return out

    base = make_batch()
    dup = duplicate_view_in_batch(base, src=0, at=1)
    check("the duplicated window is one view longer", len(dup) == len(base) + 1,
          f"{len(base)} -> {len(dup)}")
    check("the duplicate's ground truth is byte-identical",
          torch.equal(dup[0]["pts3d"], dup[1]["pts3d"]))

    class _Acc:  # loss_of_one_batch only needs the object to exist
        pass

    out = {}
    for name, b in (("base", base), ("dup", dup)):
        net.zero_grad(set_to_none=True)
        r = loss_of_one_batch(b, net, crit, _Acc(), symmetrize_batch=False, use_amp=False)
        loss, details = r["loss"]
        loss.backward()
        gnorm = sum(float(p.grad.norm()) ** 2 for p in net.parameters() if p.grad is not None) ** 0.5
        out[name] = (float(loss), gnorm, len(r["pred"]))
        print(f"       {name:5s} views={len(r['pred'])}  loss={float(loss):.6f}  |grad|={gnorm:.4e}")
        check(f"{name}: loss is finite", np.isfinite(float(loss)))
        check(f"{name}: gradients flow", gnorm > 0)

    check("the model predicted the extra view", out["dup"][2] == out["base"][2] + 1,
          f"{out['base'][2]} -> {out['dup'][2]}")
    check("the duplicate changes the loss (it is not silently dropped)",
          out["dup"][0] != out["base"][0],
          f"{out['base'][0]:.6f} vs {out['dup'][0]:.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model-cfg", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq", default="examples/002")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    structural_tests()
    if args.ckpt:
        loss_test(args.ckpt, args.model_cfg, args.device, args.seq, args.frames, args.size)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
