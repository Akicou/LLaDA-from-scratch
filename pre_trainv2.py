from model import LLaDAModel, LLaDAModelLM
from configs_llada import ActivationCheckpointingStrategy
import torch
from dataset import LLaDADatasetV2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from training_setup import TrainingSettings, build_hf_config, build_model_config, detect_device
from torch.nn.utils.rnn import pad_sequence

device = detect_device()
settings = TrainingSettings()
model_100M = build_model_config(device)
hf_configs = build_hf_config(device)

def collate_fn_stack(batch):
    out = {}

    # 1) Apilar t (escalar)        
    t_vals = [sample["t"] for sample in batch]
    if not torch.is_tensor(t_vals[0]):
        t_vals = [torch.tensor(v, dtype=torch.float32) for v in t_vals]
    out["t"] = torch.stack(t_vals, dim=0)  # [B]

    # 2) Campos de secuencia
    seq_keys = ["input_ids", "noisy_input_ids", "mask"]
    # Descubre el pad_id (puedes ajustar según tu tokenizer/modelo)
    pad_id = tokenizer.pad_token_id if "tokenizer" in globals() else 0

    for key in seq_keys:
        seqs = [sample[key] for sample in batch]
        # Asegura que son tensores
        if not torch.is_tensor(seqs[0]):
            dtype = torch.long if key != "mask" else torch.bool
            seqs = [torch.tensor(s, dtype=dtype) for s in seqs]
        pad_val = False if key == "mask" else pad_id
        padded = pad_sequence(seqs, batch_first=True, padding_value=pad_val)
        out[key] = padded  # [B, L_max]

    return out

print("Load model test")
model = LLaDAModel(model_100M, init_params=True)
model.set_activation_checkpointing(ActivationCheckpointingStrategy.one_in_two)
tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name)
hf_model = LLaDAModelLM(config=hf_configs, model=model)
print("Model test success")

dataset = LLaDADatasetV2(list(settings.dataset_paths))
dataloader = DataLoader(
    dataset,
    batch_size=settings.batch_size,
    shuffle=True,
    num_workers=settings.num_workers,
    pin_memory=settings.pin_memory,
    collate_fn=collate_fn_stack
)


optimizer = AdamW(hf_model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)

total_steps = settings.total_steps
warmup_steps = settings.warmup_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

log_every = settings.log_every
save_every = settings.save_every
output_dir = settings.checkpoint_path()

for step, batch in enumerate(dataloader, start=1):
    hf_model.train()
    optimizer.zero_grad()

    # Batch now on device
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    inp   = batch["input_ids"]#.to(device)        # [B, L]
    noisy = batch["noisy_input_ids"]#.to(device)  # [B, L]
    mask  = batch["mask"]#.to(device)             # [B, L]
    t_vals= batch["t"]#.to(device)                # [B]

    # Sanity check prints
    if step % log_every == 0 or step == 1:
        print(f"\nStep {step}")
        print("→ t samples:", t_vals[:5].tolist())
        print("→ Masked ratios:", mask.float().mean(dim=1)[:5].tolist())


    # Forward
    logits = hf_model(noisy).logits                 # [B, L, V]

    # Loss diffusion: CE only on masked tokens, weighted 1/t
    B = inp.size(0)
    total_loss = 0.0
    for i in range(B):
        ti = t_vals[i]
        mi = mask[i]
        logits_i = logits[i, mi]              # [Ni, V]
        target_i = inp[i, mi]                 # [Ni]
        ce = F.cross_entropy(logits_i, target_i, reduction="sum")
        total_loss += ce / ti

    loss = total_loss / B

    # Backward + gradient clipping 
    loss.backward()
    # Grad norm check
    grad_norm = torch.nn.utils.clip_grad_norm_(hf_model.parameters(), max_norm=1.0)
    if step % log_every == 0 or step == 1:
        print(f"→ Grad norm: {grad_norm:.4f}")

    # Optimizer & scheduler step
    optimizer.step()
    scheduler.step()

    # Logging y checkpoints
    if step % log_every == 0:
        ppl = torch.exp(loss).item()
        print(f"[Step {step:6d}/{total_steps}] loss={loss.item():.4f} ppl={ppl:.2f}")

    if step % save_every == 0:
        checkpoint_dir = output_dir / f"llada_ckpt_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save the model in transformers format
        hf_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
