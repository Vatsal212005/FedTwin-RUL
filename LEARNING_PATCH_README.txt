TwinFed-MoE LEARNING PATCH
===========================

Paste these files over the root of the current TwinFed-MoE project and
choose Replace when Windows prompts.

Changes:
- batch size: 512 -> 256
- local epochs: 1 -> 2
- FL rounds: 10 -> 40
- client participation: 50% -> 100%
- RUL target is normalized from [0,125] to [0,1] for optimization
- model RUL head uses Sigmoid to predict normalized RUL
- evaluation converts predictions back to cycle units
- every communication round is evaluated on ALL 10 fixed validation clients
- shared global layers are broadcast to all clients after each round
- personalized adapters remain local

NO REPROCESSING IS REQUIRED for this patch.
The stride-5 FD004_train.npz you already generated is still valid.

Run:
    python scripts\run_federated.py --config configs\fd004.yaml

Expected runtime on the same RTX 4060 should still be far below the original
multi-hour setup, likely in the low-single-digit-minute range based on the
8-second benchmark run. The actual first-round ETA is authoritative.
