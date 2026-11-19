#!/usr/bin/env python3
"""
extract_recon.py, but the model is loaded once and reused over many sequences.

The per-sequence output is identical to extract_recon.py's: forward_recurrent
re-initialises the recurrent state and the pose memory from parameters at i==0,
so sequences cannot leak into each other.

    python extract_multi.py --repo /app/CUT3R --model_path CKPT \
        --out_dir /app/recon_ab2 --prefix cut3r --seqs DIR [DIR ...]
"""
import argparse, os, sys, traceback
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="repo root holding demo.py")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_cfg", default=None)
    ap.add_argument("--enc_num_steps", type=int, default=None,
                    help="sobreescribe model.num_steps tras cargar (pasos del "
                         "encoder recurrente). NO se puede hacer con --model_cfg: "
                         "load_model prefiere los args embebidos en el checkpoint "
                         "y model_cfg_str solo es fallback para ckpts sin args. "
                         "Como croco.py:522 lee self.num_steps en tiempo de "
                         "ejecucion, asignar el atributo basta y los pesos no "
                         "cambian (los bloques son num_blocks, no num_steps).")
    ap.add_argument("--prime_state", action="store_true",
                    help="da un paso de calentamiento del estado en i=0 y tira "
                         "su prediccion, para que el frame 0 se puntue desde un "
                         "estado que ya paso por dec_blocks_state. Cero "
                         "parametros nuevos; cuesta una pasada de decoder en el "
                         "primer frame. Ver docs/attractor_mechanism.md.")
    ap.add_argument("--no_enc_rope", action="store_true",
                    help="apaga RoPE en los bloques del ENCODER poniendo "
                         "blk.attn.rope = None (blocks.py:'if self.rope is not "
                         "None'). Ablacion para medir cuanto cuesta que el "
                         "encoder no sepa donde esta cada parche: es la "
                         "condicion en la que ya vive el encoder recurrente de "
                         "LoopCUT3R, cuyo RecurrentBlock acepta xpos y lo "
                         "descarta. No toca el decoder, que conserva su RoPE.")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(args.model_path)

    import demo as D
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.inference import inference_recurrent
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.geometry import geotrf

    print(f"loading {args.model_path}", flush=True)
    kw = {"model_cfg_str": args.model_cfg} if args.model_cfg else {}
    model = ARCroco3DStereo.from_pretrained(args.model_path, **kw).to(args.device)
    model.eval()

    if args.no_enc_rope:
        n = 0
        for blk in model.enc_blocks:
            attn = getattr(blk, "attn", None)
            if attn is not None and getattr(attn, "rope", None) is not None:
                attn.rope = None
                n += 1
        print(f"no_enc_rope: RoPE apagado en {n}/{len(model.enc_blocks)} "
              f"bloques del encoder", flush=True)
        if n == 0:
            print("  !! ningun bloque tenia rope: este encoder ya era ciego a "
                  "la posicion, la ablacion no cambia nada", flush=True)

    if args.prime_state:
        model.prime_state = True
        print("prime_state: activado (un paso de calentamiento en i=0)", flush=True)

    if args.enc_num_steps is not None:
        antes = getattr(model, "num_steps", None)
        nb = len(getattr(model, "enc_blocks", []))
        model.num_steps = int(args.enc_num_steps)
        print(f"encoder: num_steps {antes} -> {model.num_steps}   "
              f"({nb} bloques x {model.num_steps} = {nb * model.num_steps} "
              f"aplicaciones)", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for sp in args.seqs:
        name = os.path.basename(sp.rstrip("/"))
        out = os.path.join(args.out_dir, f"{args.prefix}_{name}.npz")
        if os.path.exists(out) and os.path.getsize(out) > 0 and not os.environ.get("FORCE"):
            print(f"  skip {name} (exists)", flush=True)
            continue
        try:
            img_paths, tmpdir = D.parse_seq_path(sp)
            img_paths = [p for p in img_paths
                         if os.path.isfile(p) and
                         os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")]
            if not img_paths:
                print(f"  !! no images in {sp}", flush=True); continue
            views = D.prepare_input(img_paths, [True] * len(img_paths), args.size)
            with torch.no_grad():
                outputs, _ = inference_recurrent(views, model, args.device, verbose=False)
            preds, vws = outputs["pred"], outputs["views"]
            n = len(preds)
            pts_cam = torch.cat([p["pts3d_in_self_view"].cpu() for p in preds], 0)
            pts_world = torch.cat([p["pts3d_in_other_view"].cpu() for p in preds], 0)
            conf_cam = torch.cat([p["conf_self"].cpu() for p in preds], 0)
            conf_world = torch.cat([p["conf"].cpu() for p in preds], 0)
            poses = torch.cat(
                [pose_encoding_to_camera(p["camera_pose"].clone()).cpu() for p in preds], 0)
            pts_posed = torch.cat(
                [geotrf(poses[i:i+1], pts_cam[i:i+1]) for i in range(n)], 0)
            rgb = torch.cat(
                [0.5 * (v["img"].permute(0, 2, 3, 1).cpu() + 1.0) for v in vws], 0)
            f32 = lambda t: t.numpy().astype(np.float32)
            np.savez_compressed(
                out,
                pts_cam=f32(pts_cam), pts_world=f32(pts_world), pts_posed=f32(pts_posed),
                conf_cam=f32(conf_cam), conf_world=f32(conf_world), pose_c2w=f32(poses),
                rgb=f32(rgb),
                img_paths=np.array([str(p) for p in img_paths]),
                seq=np.array(sp), model_path=np.array(args.model_path),
                size=np.array(args.size),
            )
            print(f"  ok {name}  frames={n}  HxW={tuple(pts_cam.shape[1:3])}  "
                  f"{os.path.getsize(out)/1e6:.0f} MB", flush=True)
            if tmpdir:
                import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            print(f"  !! FAILED {name}", flush=True)
            traceback.print_exc()
    print("done", flush=True)


if __name__ == "__main__":
    main()
