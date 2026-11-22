TwinFed-MoE FAST RESEARCH PATCH

Paste these files over your current TwinFed-MoE folder and choose Replace.

Changes:
- training-window stride 1 -> 5
- balanced regime-aware whole-engine client partition
- batch size 512
- 1-layer d_model=64 Transformer
- 4 experts, Top-2 routing retained
- 1 local epoch
- 10 FL rounds
- 50% client participation per round
- CUDA mixed precision
- sampled TGC loss (256 pairs max)
- FedProx global reference copied to GPU once per client
- client timing, round timing, elapsed time and ETA

Your data/raw/cmapss folder is untouched.

After pasting:
1) python scripts\preprocess_cmapss.py --subset FD004
2) python scripts\validate_cmapss.py --subset FD004
3) python scripts\run_federated.py --config configs\fd004.yaml

Expected FD004 training windows after stride=5: about 10k-11k.
The held-out NASA test split remains 248 engines.

If CUDA OOM occurs, change batch_size from 512 to 256.
