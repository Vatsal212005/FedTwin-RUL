"""Ablation experiment launcher skeleton.

Recommended ablations:
A0 Full TwinFed-MoE
A1 No Digital Twin conditioning
A2 No MoE
A3 No personalized adapter
A4 No twin-guided consistency loss
A5 FedAvg instead of expert-wise aggregation
A6 Dense routing instead of Top-k routing
"""

if __name__ == "__main__":
    print("TODO: add config overrides and experiment logging.")
