import torch

from .losses import (
    rul_mse_loss,
    digital_twin_reconstruction_loss,
    load_balance_loss,
    twin_guided_consistency_loss,
    prepare_fedprox_reference,
    fedprox_loss,
)
from .metrics import (
    metric_dict,
    routing_entropy,
)


def _rul_cap(cfg):
    return float(
        cfg.get(
            "data",
            {},
        ).get(
            "rul_cap",
            125.0,
        )
    )


def _zero(model):
    return next(
        model.parameters()
    ).new_tensor(0.0)


def train_local(
    model,
    loader,
    optimizer,
    device,
    cfg,
    global_state=None,
):
    model.train()

    total = 0.0
    total_n = 0

    util_sum = None
    util_batches = 0

    use_amp = (
        device.type == "cuda"
        and cfg["training"].get(
            "amp",
            True,
        )
    )

    amp_dtype = (
        torch.bfloat16
        if cfg["training"].get(
            "amp_dtype",
            "float16",
        ) == "bfloat16"
        else torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            use_amp
            and amp_dtype
            == torch.float16
        ),
    )

    prox_ref = (
        prepare_fedprox_reference(
            global_state,
            device,
        )
    )

    tgc_pairs = int(
        cfg["training"].get(
            "tgc_max_pairs",
            256,
        )
    )

    rul_cap = _rul_cap(
        cfg
    )

    use_dt = bool(
        getattr(
            model,
            "use_digital_twin",
            True,
        )
    )

    use_moe = bool(
        getattr(
            model,
            "use_moe",
            True,
        )
    )

    router_mode = str(
        getattr(
            model,
            "router_mode",
            "twin",
        )
    )

    for _ in range(
        cfg["training"][
            "local_epochs"
        ]
    ):
        for (
            x,
            twin_aux,
            y,
        ) in loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            twin_aux = (
                twin_aux.to(
                    device,
                    non_blocking=True,
                )
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            y_normalized = (
                torch.clamp(
                    y / rul_cap,
                    min=0.0,
                    max=1.0,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                out = model(
                    x,
                    twin_aux,
                    use_personal_adapter=True,
                )

                l_rul = (
                    rul_mse_loss(
                        out["prediction"],
                        y_normalized,
                    )
                )

                if use_dt:
                    l_dt = (
                        digital_twin_reconstruction_loss(
                            out[
                                "reconstruction"
                            ],
                            out[
                                "sensor_summary"
                            ],
                        )
                    )
                else:
                    l_dt = _zero(
                        model
                    )

                if use_moe:
                    l_bal = (
                        load_balance_loss(
                            out[
                                "dense_gate"
                            ]
                        )
                    )
                else:
                    l_bal = _zero(
                        model
                    )

                if (
                    use_dt
                    and use_moe
                    and router_mode
                    == "twin"
                    and cfg[
                        "training"
                    ].get(
                        "lambda_tgc",
                        0.0,
                    ) > 0
                ):
                    l_tgc = (
                        twin_guided_consistency_loss(
                            out[
                                "twin_state"
                            ],
                            out[
                                "dense_gate"
                            ],
                            max_pairs=(
                                tgc_pairs
                            ),
                        )
                    )
                else:
                    l_tgc = _zero(
                        model
                    )

                l_prox = (
                    fedprox_loss(
                        model,
                        prox_ref,
                        cfg[
                            "training"
                        ].get(
                            "prox_mu",
                            0.0,
                        ),
                    )
                )

                loss = (
                    l_rul
                    + cfg[
                        "training"
                    ].get(
                        "lambda_dt",
                        0.0,
                    ) * l_dt
                    + cfg[
                        "training"
                    ].get(
                        "lambda_bal",
                        0.0,
                    ) * l_bal
                    + cfg[
                        "training"
                    ].get(
                        "lambda_tgc",
                        0.0,
                    ) * l_tgc
                    + l_prox
                )

            if scaler.is_enabled():
                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                optimizer.step()

            bs = x.size(0)

            total += (
                float(
                    loss.detach()
                )
                * bs
            )

            total_n += bs

            util = (
                out[
                    "expert_utilization"
                ]
                .detach()
                .cpu()
            )

            util_sum = (
                util
                if util_sum is None
                else util_sum + util
            )

            util_batches += 1

    return {
        "loss": (
            total
            / max(
                1,
                total_n,
            )
        ),
        "expert_utilization": (
            util_sum
            / max(
                1,
                util_batches,
            )
        ),
    }


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    cfg,
    use_personal_adapter: bool = True,
    collect_routing: bool = False,
):
    model.eval()

    preds = []
    targets = []

    dense_gates = []
    sparse_gates = []
    twin_states = []

    use_amp = (
        device.type == "cuda"
    )

    rul_cap = _rul_cap(
        cfg
    )

    for (
        x,
        twin_aux,
        y,
    ) in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        twin_aux = (
            twin_aux.to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            out = model(
                x,
                twin_aux,
                use_personal_adapter=(
                    use_personal_adapter
                ),
            )

        pred_cycles = (
            out[
                "prediction"
            ]
            .float()
            .cpu()
            * rul_cap
        )

        preds.append(
            pred_cycles
        )

        targets.append(
            y.float().cpu()
        )

        if collect_routing:
            dense_gates.append(
                out[
                    "dense_gate"
                ]
                .float()
                .cpu()
            )

            sparse_gates.append(
                out[
                    "sparse_gate"
                ]
                .float()
                .cpu()
            )

            twin_states.append(
                out[
                    "twin_state"
                ]
                .float()
                .cpu()
            )

    result = {
        "pred": torch.cat(
            preds
        ),
        "target": torch.cat(
            targets
        ),
    }

    if collect_routing:
        dense = torch.cat(
            dense_gates
        )

        sparse = torch.cat(
            sparse_gates
        )

        result.update({
            "dense_gate": dense,
            "sparse_gate": sparse,
            "twin_state": torch.cat(
                twin_states
            ),
            "expert_utilization": (
                sparse.mean(
                    dim=0
                )
            ),
            "routing_entropy": (
                routing_entropy(
                    dense
                )
            ),
        })

    return result


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    cfg,
    use_personal_adapter: bool = True,
):
    result = predict(
        model,
        loader,
        device,
        cfg,
        use_personal_adapter=(
            use_personal_adapter
        ),
        collect_routing=False,
    )

    return metric_dict(
        result["pred"],
        result["target"],
    )
