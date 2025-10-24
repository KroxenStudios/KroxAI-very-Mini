from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from typing import List, Dict, Any, Iterable
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler

from .tokenizer import SimpleTokenizer
from .hf_tokenizer import HFTokenizer
from .data import load_qa_json, build_sequences
from .torch_model import TorchTransformerLM
from .configs import get_preset


class SeqDataset(Dataset):
    def __init__(self, seqs: List[torch.Tensor], weights: List[float] | None = None):
        self.seqs = seqs
        self.weights = weights or [1.0] * len(seqs)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def pad_collate(batch: List[torch.Tensor], pad_id: int):
    T = max(x.size(0) for x in batch)
    out = torch.full((len(batch), T), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, : x.size(0)] = x
    return out


def train_one_epoch(model: TorchTransformerLM, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, pad_id: int, amp: bool = False, scaler: torch.cuda.amp.GradScaler | None = None, grad_accum_steps: int = 1, scheduler = None, max_batches: int | None = None):
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    step = 0
    processed_batches = 0
    for i, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=True)
        # inputs are all tokens except last; targets are next tokens
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        # If sequence exceeds model context, crop both inp and tgt to last max_len steps
        max_len = getattr(model, 'max_len', None)
        if max_len is not None and inp.size(1) > max_len:
            inp = inp[:, -max_len:]
            tgt = tgt[:, -max_len:]
        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(inp)
                # Align target length to logits time dimension (may be shorter due to internal truncation)
                if logits.size(1) != tgt.size(1):
                    tgt = tgt[:, -logits.size(1):]
                loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            (scaler.scale(loss) / grad_accum_steps).backward()
        else:
            logits = model(inp)
            if logits.size(1) != tgt.size(1):
                tgt = tgt[:, -logits.size(1):]
            loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            (loss / grad_accum_steps).backward()
        total_loss += float(loss.item())
        if i % grad_accum_steps == 0:
            if amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            step += 1
        processed_batches += 1
        # In streaming mode, cap the number of processed batches to keep epochs finite
        if max_batches is not None and processed_batches >= max_batches:
            break
    # Average over processed mini-batches to keep metric comparable across modes
    return total_loss / max(1, processed_batches)


@torch.inference_mode()
def evaluate(model: TorchTransformerLM, loader: DataLoader, device: torch.device, pad_id: int):
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        max_len = getattr(model, 'max_len', None)
        if max_len is not None and inp.size(1) > max_len:
            inp = inp[:, -max_len:]
            tgt = tgt[:, -max_len:]
        logits = model(inp)
        if logits.size(1) != tgt.size(1):
            tgt = tgt[:, -logits.size(1):]
        loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        total += float(loss.item())
        n += 1
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs='+', help="Path(s) or glob(s) to training data shards (ptseqs). Comma-separated or repeated. Optionally add :weight for sampling weight, e.g. data/wiki-*.pt:0.7 data/chat-*.pt:0.3")
    p.add_argument("--val-data", type=str, default=None, help="Optional: path/glob for validation shards in streaming mode (ptseqs)")
    p.add_argument("--val-items-per-shard", type=int, default=0, help="If set, reserve N items per shard for validation in streaming mode (default: 0)")
    p.add_argument("--more-data", nargs="*", default=None, help="Additional data files to include for training (same format as --data-format)")
    p.add_argument("--data-format", type=str, default="qna", choices=["qna","chatjsonl","plainjsonl","ptseqs"], help="Dataset format: qna (JSON array with q/a), chatjsonl (JSONL with messages), plainjsonl (JSONL with {text}), or ptseqs (torch.save payload with 'seqs')")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--preset", type=str, default=None, help="Model size preset: tiny|small|base|large")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--val-split", type=float, default=0.05, help="Fraction of data for validation")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none","cosine"], help="LR scheduler")
    p.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio of total steps (if scheduler != none)")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--checkpoint", type=str, default="kroxai_trained.pt")
    p.add_argument("--save-best", action="store_true", help="Save only the best checkpoint by val loss")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only)")
    p.add_argument("--use-weights", action="store_true", help="Use QAItem.weight for weighted sampling in training")
    # Streaming options for ptseqs
    p.add_argument("--streaming", action="store_true", help="Enable streaming dataset for ptseqs shards (iterate shard-by-shard, low memory)")
    p.add_argument("--steps-per-epoch", type=int, default=0, help="When streaming, number of optimizer steps per epoch for scheduler. If 0, scheduler will be disabled unless dataset length is known.")
    p.add_argument("--stream-no-shuffle-shards", action="store_true", help="When streaming, do not shuffle shard order between epochs")
    p.add_argument("--stream-no-shuffle-within-shard", action="store_true", help="When streaming, do not shuffle sequence order within a shard")
    # Tokenizer options
    p.add_argument("--tokenizer", type=str, default="simple", choices=["simple", "hf"], help="Tokenizer backend")
    p.add_argument("--tokenizer-path", type=str, default=None, help="Path to HF tokenizer (.json or .model/.spm)")
    p.add_argument("--pad-id", type=int, default=None, help="Pad token id for HF tokenizer (required if hf has no pad)")
    p.add_argument("--bos-id", type=int, default=None, help="BOS token id for HF tokenizer if available")
    p.add_argument("--eos-id", type=int, default=None, help="EOS token id for HF tokenizer if available")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    # Tokenizer selection
    if args.tokenizer == "hf":
        if not args.tokenizer_path:
            raise SystemExit("--tokenizer hf requires --tokenizer-path to a tokenizer file")
        tk = HFTokenizer(args.tokenizer_path)
        # optionally set special ids on instance for build_sequences compatibility
        if args.pad_id is not None:
            setattr(tk, "PAD", int(args.pad_id))
        if args.bos_id is not None:
            setattr(tk, "BOS", int(args.bos_id))
        if args.eos_id is not None:
            setattr(tk, "EOS", int(args.eos_id))
        # derive pad id
        pad_id = getattr(tk, "PAD", None)
        if pad_id is None:
            raise SystemExit("HF tokenizer requires --pad-id to be set (no PAD token found)")
        pad_id = int(pad_id)
    else:
        tk = SimpleTokenizer()
        pad_id = tk.PAD
    # Data loading: QNA JSON or Chat JSONL
    def _encode_text_to_ids(text: str) -> List[int]:
        add_bos = hasattr(tk, "BOS")
        add_eos = hasattr(tk, "EOS")
        return tk.encode(text, add_bos=add_bos, add_eos=add_eos)

    def load_chatjsonl_to_sequences(path: str) -> tuple[List[List[int]], List[float]]:
        seqs: List[List[int]] = []
        weights: List[float] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj: Dict[str, Any] = json.loads(line)
                except Exception:
                    continue
                msgs = obj.get("messages") or []
                if not isinstance(msgs, list) or not msgs:
                    continue
                # Build a simple prompt: optional system header + last user question and assistant answer
                sys_txt = None
                last_user = None
                last_assistant = None
                for m in msgs:
                    role = (m.get("role") or "").lower()
                    content = (m.get("content") or "").strip()
                    if not content:
                        continue
                    if role == "system":
                        sys_txt = content
                    elif role == "user":
                        last_user = content
                    elif role == "assistant":
                        last_assistant = content
                if not last_user or not last_assistant:
                    continue
                parts: List[str] = []
                if sys_txt:
                    parts.append(f"System: {sys_txt}\n")
                parts.append("Q: ")
                parts.append(last_user)
                parts.append("\nA: ")
                parts.append(last_assistant)
                text = "".join(parts)
                seqs.append(_encode_text_to_ids(text))
                weights.append(float(obj.get("weight", 1.0)))
        return seqs, weights

    def load_plainjsonl_to_sequences(path: str) -> tuple[List[List[int]], List[float]]:
        """Load JSONL where each line contains {"text": "...", optional "weight"} and encode directly for LM training."""
        seqs: List[List[int]] = []
        weights: List[float] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj: Dict[str, Any] = json.loads(line)
                except Exception:
                    continue
                text = (obj.get("text") or obj.get("content") or "").strip()
                if not text:
                    continue
                seqs.append(_encode_text_to_ids(text))
                weights.append(float(obj.get("weight", 1.0)))
        return seqs, weights

    if args.data_format == "qna":
        items = load_qa_json(args.data)
        if args.more_data:
            for extra in args.more_data:
                items.extend(load_qa_json(extra))
        np_seqs = build_sequences(items, tk)
        sample_weights = [float(it.weight) for it in items]
    elif args.data_format == "chatjsonl":
        # chatjsonl
        seq_list, weights_list = load_chatjsonl_to_sequences(args.data)
        if args.more_data:
            for extra in args.more_data:
                s2, w2 = load_chatjsonl_to_sequences(extra)
                seq_list.extend(s2)
                weights_list.extend(w2)
        np_seqs = seq_list
        sample_weights = weights_list
    elif args.data_format == "plainjsonl":
        # plainjsonl
        seq_list, weights_list = load_plainjsonl_to_sequences(args.data)
        if args.more_data:
            for extra in args.more_data:
                s2, w2 = load_plainjsonl_to_sequences(extra)
                seq_list.extend(s2)
                weights_list.extend(w2)
        np_seqs = [torch.tensor(s, dtype=torch.long).numpy() for s in seq_list]
        sample_weights = weights_list
    else:
        # ptseqs: support single .pt file or a directory / glob of shard files
        # Parse multiple data sources with optional weights
        import glob
        data_specs = []
        for darg in args.data:
            for part in darg.split(','):
                part = part.strip()
                if not part:
                    continue
                if ':' in part:
                    pat, w = part.rsplit(':', 1)
                    try:
                        w = float(w)
                    except Exception:
                        w = 1.0
                else:
                    pat, w = part, 1.0
                files = sorted(glob.glob(pat)) if any(ch in pat for ch in '*?[') else ([pat] if os.path.isfile(pat) else [])
                if not files:
                    # try directory
                    if os.path.isdir(pat):
                        files = [os.path.join(pat, f) for f in os.listdir(pat) if f.endswith('.pt')]
                if not files:
                    print(f"[torch_train] Warnung: keine Dateien für Muster {pat}")
                    continue
                data_specs.append({'files': files, 'weight': float(w)})
        if not data_specs:
            raise SystemExit(f"ptseqs: keine .pt shards gefunden für '{args.data}'")
        total_files = sum(len(ds['files']) for ds in data_specs)
        print(f"[torch_train] Found {total_files} shard(s) from {len(data_specs)} source(s)")
        # If streaming is enabled, create an IterableDataset; otherwise, load into memory
        ds_tk: Dict[str, Any] = {}
        # Peek first shard for tokenizer config to adopt pad_id
        first_file = next((f for ds in data_specs for f in ds['files']), None)
        if first_file:
            sd0 = torch.load(first_file, map_location='cpu')
            ds_tk = sd0.get('tokenizer') or {}
            if not isinstance(ds_tk, dict):
                ds_tk = {}
            if 'pad_id' in ds_tk:
                pad_id = int(ds_tk.get('pad_id', pad_id))
            del sd0

    if args.streaming:
            if args.use_weights:
                print("[torch_train] Hinweis: --use-weights wird im Streaming-Modus ignoriert.")
            shuffle_shards = not args.stream_no_shuffle_shards
            shuffle_within = not args.stream_no_shuffle_within_shard

            class MixedPtShardIterableDataset(torch.utils.data.IterableDataset):
                def __init__(self, data_specs, shuffle_shards=True, shuffle_within_shard=True, val_items_per_shard=0, mode='train'):
                    super().__init__()
                    self.data_specs = data_specs
                    self.shuffle_shards = shuffle_shards
                    self.shuffle_within = shuffle_within_shard
                    self.val_items_per_shard = val_items_per_shard
                    self.mode = mode  # 'train' or 'val'

                def __iter__(self) -> Iterable[torch.Tensor]:
                    worker_info = torch.utils.data.get_worker_info()
                    base_seed = torch.initial_seed() if worker_info is None else worker_info.seed
                    rng = random.Random(base_seed)
                    # Build a flat list of (files, weight) for sampling
                    all_files = []
                    weights = []
                    for ds in self.data_specs:
                        for f in ds['files']:
                            all_files.append(f)
                            weights.append(ds['weight'])
                    if not all_files:
                        return
                    # Normalize weights
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    # For validation: yield only first N items per shard
                    if self.mode == 'val' and self.val_items_per_shard > 0:
                        for f in all_files:
                            sd = torch.load(f, map_location='cpu')
                            seq_tensors = sd.get('seqs') or []
                            n = min(self.val_items_per_shard, len(seq_tensors))
                            for i in range(n):
                                t = seq_tensors[i]
                                if not isinstance(t, torch.Tensor):
                                    t = torch.tensor(t, dtype=torch.long)
                                yield t
                            del sd
                        return
                    # Training: sample shards by weight, shuffle if needed
                    files = list(all_files)
                    wts = list(weights)
                    if self.shuffle_shards:
                        idxs = list(range(len(files)))
                        rng.shuffle(idxs)
                        files = [files[i] for i in idxs]
                        wts = [wts[i] for i in idxs]
                    while True:
                        # Sample a file index by weights
                        idx = rng.choices(range(len(files)), weights=wts, k=1)[0]
                        f = files[idx]
                        sd = torch.load(f, map_location='cpu')
                        seq_tensors = sd.get('seqs') or []
                        indices = list(range(len(seq_tensors)))
                        if self.shuffle_within:
                            rng.shuffle(indices)
                        for i in indices:
                            # If val_items_per_shard > 0, skip first N for training
                            if self.val_items_per_shard > 0 and i < self.val_items_per_shard:
                                continue
                            t = seq_tensors[i]
                            if not isinstance(t, torch.Tensor):
                                t = torch.tensor(t, dtype=torch.long)
                            yield t
                        del sd

            train_ds = MixedPtShardIterableDataset(data_specs, shuffle_shards, shuffle_within, val_items_per_shard=args.val_items_per_shard, mode='train')
            val_loader = None
            if args.val_data or args.val_items_per_shard > 0:
                # Validation from separate pattern or reserved items per shard
                val_specs = []
                if args.val_data:
                    import glob, os
                    for part in args.val_data.split(','):
                        part = part.strip()
                        if not part:
                            continue
                        files = sorted(glob.glob(part)) if any(ch in part for ch in '*?[') else ([part] if os.path.isfile(part) else [])
                        if not files:
                            if os.path.isdir(part):
                                files = [os.path.join(part, f) for f in os.listdir(part) if f.endswith('.pt')]
                        if not files:
                            print(f"[torch_train] Warnung: keine Dateien für val-Muster {part}")
                            continue
                        val_specs.append({'files': files, 'weight': 1.0})
                else:
                    val_specs = data_specs
                val_ds = MixedPtShardIterableDataset(val_specs, shuffle_shards=False, shuffle_within_shard=False, val_items_per_shard=args.val_items_per_shard, mode='val')
                val_loader = DataLoader(
                    val_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=lambda b: pad_collate(b, pad_id=pad_id),
                    pin_memory=(device.type == "cuda"),
                )
            sample_weights = None
            # Build DataLoader for iterable dataset
            dl = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda b: pad_collate(b, pad_id=pad_id),
                pin_memory=(device.type == "cuda"),
            )
            # In streaming mode, epochs are defined by steps-per-epoch if provided
            max_batches_per_epoch = None
            if args.steps_per_epoch and args.steps_per_epoch > 0:
                # steps-per-epoch counts optimizer steps; convert to minibatches with grad accumulation
                max_batches_per_epoch = max(1, args.steps_per_epoch) * max(1, args.grad_accum_steps)
    else:
        # Non-streaming: load all shards into memory
        np_seqs: List = []
        sample_weights: List[float] = []
        total_seq_count = 0
        # Flatten all files from all data_specs
        all_files = [f for ds in data_specs for f in ds['files']]
        for sf in all_files:
            sd = torch.load(sf, map_location='cpu')
            seq_tensors = sd.get('seqs') or []
            if not isinstance(seq_tensors, list) or not seq_tensors:
                print(f"[torch_train] Warnung: shard {sf} enthält keine 'seqs', übersprungen")
                continue
            for t in seq_tensors:
                if isinstance(t, torch.Tensor):
                    np_seqs.append(t.cpu().numpy())
                else:
                    np_seqs.append(t)
            ws = sd.get('weights') or []
            if isinstance(ws, list) and len(ws) == len(seq_tensors):
                sample_weights.extend([float(w) for w in ws])
            else:
                sample_weights.extend([1.0] * len(seq_tensors))
            total_seq_count += len(seq_tensors)
        if not np_seqs:
            raise SystemExit("ptseqs: no sequences found in shards")
        if args.tokenizer == "simple" and (ds_tk.get('type') == 'hf'):
            print("[torch_train] Hinweis: Dataset wurde mit HF-Tokenizer erstellt. Erwäge --tokenizer hf --tokenizer-path ...")
        print(f"[torch_train] Insgesamt Sequenzen geladen: {total_seq_count}")
        # Convert to Torch tensors
        seqs = [torch.tensor(s, dtype=torch.long) for s in np_seqs]
        ds = SeqDataset(seqs, sample_weights)
        # Split into train/val
        val_size = max(1, int(len(ds) * args.val_split)) if len(ds) > 1 else 0
        train_size = len(ds) - val_size
        if val_size > 0:
            train_ds, val_ds = random_split(ds, [train_size, val_size])
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: pad_collate(b, pad_id=pad_id), pin_memory=(device.type == "cuda"))
        else:
            train_ds, val_loader = ds, None
        # Configure sampler: weighted sampling if requested
        sampler = None
        shuffle = True
        if args.use_weights:
            if hasattr(train_ds, 'indices') and isinstance(train_ds.indices, list):
                idxs = train_ds.indices
                weights_tensor = torch.tensor([ds.weights[i] for i in idxs], dtype=torch.double)
            else:
                weights_tensor = torch.tensor(getattr(train_ds, 'weights', ds.weights), dtype=torch.double)
            weights_tensor = torch.clamp(weights_tensor, min=1e-8)
            sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
            shuffle = False
        dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=lambda b: pad_collate(b, pad_id=pad_id),
            pin_memory=(device.type == "cuda"),
        )
    # At this point, dl and val_loader are defined in both streaming and non-streaming branches

    if args.preset:
        cfg = get_preset(args.preset)
        # allow explicit override of max_len
        dim, layers, heads = cfg.dim, cfg.n_layers, cfg.n_heads
        ff = cfg.ff_hidden
        max_len = args.max_len if args.max_len and args.max_len != cfg.max_len else cfg.max_len
    else:
        dim, layers, heads, ff, max_len = args.dim, args.layers, args.heads, args.ff, args.max_len

    model = TorchTransformerLM(vocab_size=tk.vocab_size, dim=dim, n_layers=layers, n_heads=heads, ff_hidden=ff, max_len=max_len, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Determine steps-per-epoch for scheduler and training loop when streaming
    max_batches_per_epoch = locals().get('max_batches_per_epoch', None)

    # Scheduler with warmup
    if args.scheduler == "cosine":
        if args.streaming:
            if args.steps_per_epoch and args.steps_per_epoch > 0:
                total_steps = max(1, args.steps_per_epoch) * max(1, args.epochs)
                warmup_steps = max(1, int(total_steps * args.warmup_ratio))
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            else:
                print("[torch_train] Scheduler deaktiviert: --steps-per-epoch nicht gesetzt im Streaming-Modus.")
                scheduler = None
        else:
            total_steps = math.ceil(len(dl) / max(1, args.grad_accum_steps)) * max(1, args.epochs)
            warmup_steps = max(1, int(total_steps * args.warmup_ratio))
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        scheduler = None

    best_val = float('inf')
    best_path = args.checkpoint if args.checkpoint.endswith('.pt') else args.checkpoint.replace('.npz', '.pt')

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            dl,
            opt,
            device,
            pad_id=pad_id,
            amp=use_amp,
            scaler=scaler,
            grad_accum_steps=max(1, args.grad_accum_steps),
            scheduler=scheduler,
            max_batches=max_batches_per_epoch,
        )
        msg = f"epoch {epoch} train_loss: {loss:.4f}"
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, pad_id=pad_id)
            msg += f", val_loss: {val_loss:.4f}"
            if args.save_best and val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "config": {
                    "vocab_size": tk.vocab_size,
                    "dim": dim,
                    "n_layers": layers,
                    "n_heads": heads,
                    "ff_hidden": ff,
                    "max_len": max_len,
                    "device": str(device),
                    "amp": use_amp,
                    "tokenizer": {
                        "type": args.tokenizer,
                        "path": args.tokenizer_path,
                        "pad_id": pad_id,
                        "bos_id": getattr(tk, "BOS", None) if args.tokenizer == "hf" else SimpleTokenizer.BOS,
                        "eos_id": getattr(tk, "EOS", None) if args.tokenizer == "hf" else SimpleTokenizer.EOS,
                    }
                }, "train_hparams": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "grad_accum_steps": args.grad_accum_steps,
                    "val_split": args.val_split,
                }}, best_path)
                msg += f" (saved best -> {best_path})"
        print(msg)

    # Save final if not save-best or no val set
    if not args.save_best or val_loader is None:
        torch.save({"model": model.state_dict(), "config": {
            "vocab_size": tk.vocab_size,
            "dim": dim,
            "n_layers": layers,
            "n_heads": heads,
            "ff_hidden": ff,
            "max_len": max_len,
            "device": str(device),
            "amp": use_amp,
            "tokenizer": {
                "type": args.tokenizer,
                "path": args.tokenizer_path,
                "pad_id": pad_id,
                "bos_id": getattr(tk, "BOS", None) if args.tokenizer == "hf" else SimpleTokenizer.BOS,
                "eos_id": getattr(tk, "EOS", None) if args.tokenizer == "hf" else SimpleTokenizer.EOS,
            }
        }, "train_hparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_accum_steps": args.grad_accum_steps,
            "val_split": args.val_split,
        }}, best_path)
        print(f"Saved checkpoint to {best_path}")


if __name__ == "__main__":
    main()
