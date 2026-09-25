"""Fine-tune laya-multilingual (RLCD + supervised CE) untuk id/jv/sun/en.

Adaptasi dari notebook resmi laya_finetune_typed_decisions_2xT4_kaggle.ipynb:
- 1 GPU (RTX 5050 8 GB / T4 / apapun) - DDP dihilangkan, grad-accum dinaikkan
  agar effective batch tetap 64 sekuens (8 x 8).
- bf16 autocast (default; tanpa GradScaler). --amp fp16 memakai GradScaler untuk
  GPU pra-Ampere.
- Optimizer 8-bit opsional (--optim adamw8bit) bila VRAM sempit.
- Hold-out kalibrasi + fit temperatur per tipe pertanyaan, identik notebook.

Penggunaan:
  python scripts/04_train.py --out runs/laya-idjvsuen-v1
  python scripts/04_train.py --out runs/... --resume   # lanjut dari checkpoint_latest
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_LEN = 768
HEAD_MAX_LEN = 512


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def save_checkpoint(model, tok, cfg, out_dir, extra_meta=None):
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if extra_meta:
        with open(os.path.join(out_dir, "checkpoint_meta.json"), "w") as f:
            json.dump(extra_meta, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convaiinnovations/laya-multilingual")
    p.add_argument("--data", default=os.path.join(ROOT, "data", "processed", "train_items.pt"))
    p.add_argument("--out", default=os.path.join(ROOT, "runs", "laya-idjvsuen-v1"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--lr-encoder", type=float, default=2.5e-5)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--amp", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    p.add_argument("--calib-max", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, proper_reward
    from hflocal import local_snapshot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("PERINGATAN: CUDA tidak tersedia - training di CPU akan sangat lambat.")

    model_dir = local_snapshot(args.model, ROOT)
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg.update({"gradient_checkpointing": True, "max_tokens_per_batch": 4096,
                "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN, "amp_dtype": args.amp})

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    start_epoch = 0
    latest = os.path.join(args.out, "checkpoint_latest")
    if args.resume and os.path.exists(os.path.join(latest, "model.safetensors")):
        model.load_state_dict(load_file(os.path.join(latest, "model.safetensors")), strict=True)
        with open(os.path.join(latest, "checkpoint_meta.json")) as f:
            start_epoch = json.load(f).get("epoch", 0)
        print(f"resume dari epoch {start_epoch}")

    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    all_items = torch.load(args.data, weights_only=False)
    CALIB_MAX = args.calib_max
    order = list(range(len(all_items)))
    random.Random(args.seed).shuffle(order)
    n_calib = min(CALIB_MAX, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    print(f"train: {len(train_items)} | kalibrasi (hold-out): {len(calib_items)}")

    EPOCHS = args.epochs
    MICRO_BATCH = args.micro_batch
    GRAD_ACCUM = args.grad_accum
    GROUP_SIZE = args.group_size
    SIGMA_START, SIGMA_END = 0.4, 0.1
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    use_scaler = args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler and device.type == "cuda")

    enc_params = [p_ for n_, p_ in model.named_parameters() if "encoder." in n_]
    head_params = [p_ for n_, p_ in model.named_parameters() if "encoder." not in n_]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit([
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW([
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ], weight_decay=0.01)

    updates_per_epoch = max(1, len(train_items) // (MICRO_BATCH * GRAD_ACCUM))
    total_updates = updates_per_epoch * max(1, EPOCHS - start_epoch)
    warmup_steps = min(args.warmup, total_updates // 10) if total_updates > 10 else 0
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates - warmup_steps), eta_min=1e-6)
    sched = (torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps)),
        cos,
    ], milestones=[max(1, warmup_steps)]) if warmup_steps > 0 else cos)

    print(f"mulai training: {EPOCHS} epoch | effective batch {MICRO_BATCH * GRAD_ACCUM} | "
          f"{updates_per_epoch} update/epoch | amp {args.amp} | {args.optim}")
    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        random.seed(42 + epoch)
        random.shuffle(train_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(train_items), MICRO_BATCH):
            chunk = train_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            try:
                batch = collate_train_batch(chunk, tok.pad_token_id)
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=device.type == "cuda"):
                    logits, act = model(
                        batch["input_ids"].to(device), batch["attention_mask"].to(device),
                        batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                        batch["qtype"].to(device))
            except torch.OutOfMemoryError:
                # Batch terpanjang + fragmentasi bisa melewati 8 GB di laptop GPU.
                # Buang gradian batch ini dan lanjut - satu batch dari 42k tidak signifikan.
                print(f"  [OOM] lewati micro-batch di {b_idx} ({time.time()-t0:.0f}s)", flush=True)
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                accum_step = 0
                continue

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                                  w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            if use_scaler:
                try:
                    scaler.scale(loss).backward()
                except torch.OutOfMemoryError:
                    print(f"  [OOM-backward] lewati micro-batch di {b_idx}", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    accum_step = 0
                    continue
            else:
                try:
                    loss.backward()
                except torch.OutOfMemoryError:
                    print(f"  [OOM-backward] lewati micro-batch di {b_idx}", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    accum_step = 0
                    continue
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(train_items):
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  epoch {epoch + 1}/{EPOCHS} | step {n_batches} | "
                      f"loss {loss.item() * GRAD_ACCUM:.4f} | reward {r.mean().item():.3f} | "
                      f"lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s", flush=True)

        print(f"=== epoch {epoch + 1}/{EPOCHS} selesai | avg loss {epoch_loss / max(1, n_batches):.4f} "
              f"| {(time.time() - t0) / 60:.1f} menam ===", flush=True)
        save_checkpoint(model, tok, cfg, latest, extra_meta={
            "epoch": epoch + 1, "total_epochs": EPOCHS,
            "avg_loss": epoch_loss / max(1, n_batches)})

    # ---- kalibrasi temperatur pada hold-out ----
    print("\nfit temperatur kalibrasi pada hold-out ...")
    model.eval()
    calib_preds = []
    with torch.no_grad():
        for c in range(0, len(calib_items), 16):
            cb = collate_train_batch(calib_items[c:c + 16], tok.pad_token_id)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                l_sub, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device),
                                 cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                                 cb["qtype"].to(device))
            l_np = l_sub.float().cpu().numpy()
            for rr, it in enumerate(calib_items[c:c + 16]):
                k = len(it["markers"])
                calib_preds.append((it["qtype"], l_np[rr, :k], it["target"]))

    fitted = [1.0, 1.0, 1.0]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
        if sel:
            fitted[qt] = fit_one_temp(sel)
    print("temperatur (choice, score, noul):", [round(t, 3) for t in fitted])

    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-multilingual-idjvsuen-v1"
    cfg["temperature"] = fitted
    cfg.pop("temperature_by_options", None)
    cfg["training"] = {"base": "convaiinnovations/laya-multilingual",
                       "epochs": EPOCHS, "languages": ["id", "jv", "sun", "en"],
                       "tasks": ["intent (MASSIVE)", "sentiment (NusaX)"],
                       "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_checkpoint(model, tok, cfg, args.out, extra_meta={"epoch": EPOCHS, "avg_loss": None,
                                                           "temperature": fitted})
    print(f"model final tersimpan: {args.out}")


if __name__ == "__main__":
    main()
