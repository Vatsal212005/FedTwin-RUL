import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    args = parser.parse_args()

    path = Path(
        args.checkpoint
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    print(
        f"Checkpoint: {path}"
    )

    print(
        f"Round: "
        f"{checkpoint['round']}"
    )

    print(
        "Validation metrics:"
    )

    for key, value in sorted(
        checkpoint[
            "metrics"
        ].items()
    ):
        if isinstance(
            value,
            (int, float),
        ):
            print(
                f"  {key}: "
                f"{value}"
            )

    print(
        f"Stored client adapters: "
        f"{len(checkpoint.get('client_adapter_states', {}))}"
    )

    print(
        f"Stored client split records: "
        f"{len(checkpoint.get('client_splits', []))}"
    )


if __name__ == "__main__":
    main()
