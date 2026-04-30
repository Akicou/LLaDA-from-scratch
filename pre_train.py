from model import LLaDAModel, LLaDAModelLM
from configs_llada import ActivationCheckpointingStrategy
import torch
from dataset import LLaDADataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from training_setup import TrainingSettings, build_hf_config, build_model_config, detect_device

# We will train and iterate from the base with a 100M parameter model to test and then scale to the 1B model.
# Ok for training we should use ModelConfig not LLaDAConfig
device = detect_device()
settings = TrainingSettings()
model_100M = build_model_config(device)
hf_configs = build_hf_config(device)

print("Load model test")
model = LLaDAModel(model_100M, init_params=True)
model.set_activation_checkpointing(ActivationCheckpointingStrategy.one_in_two)
tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name)
hf_model = LLaDAModelLM(config=hf_configs, model=model)
print("Model test success")

dataset = LLaDADataset(list(settings.dataset_paths))
dataloader = DataLoader(
    dataset,
    batch_size=settings.batch_size,
    shuffle=True,
    num_workers=settings.num_workers,
    pin_memory=settings.pin_memory
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
    inp   = batch["input_ids"].to(device)        # [B, L]
    noisy = batch["noisy_input_ids"].to(device)  # [B, L]
    mask  = batch["mask"].to(device)             # [B, L]
    t_vals= batch["t"].to(device)                # [B]

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

        #ckpt = {
        #    "step": step,
        #    "model_state": model.state_dict(),
        #    "opt_state":   optimizer.state_dict(),
        #    "sched_state": scheduler.state_dict(),
        #}
        #torch.save(ckpt, output_dir / f"llada_ckpt_{step:06d}.pt")

"""
According to the LLaDA paper, the training loop should:

- Sample t ∼ Uniform(0,1) for each chunk. ✅
- Mask with mask = Bernoulli(t). ✅
- Calculate the cross-entropy loss only on masked tokens and divide by t. (sum) ✅
- Use AdamW (wd=0.1) and the Warmup–Stable–Decay scheduler. ✅ (It should be noted that we do not use a scheduler that changes the section based on step * batch_size * seq_len as in the paper)
- Keep max_seq_length = 4096 and the 1% chunking variable in [1,4096]. ✅ (It is already done with LLaDADataset)

https://arxiv.org/pdf/2502.09992 "Large Language Diffusion Models"
"""