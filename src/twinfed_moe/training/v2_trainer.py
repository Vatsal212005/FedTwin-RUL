import torch
import torch.nn.functional as F

from .losses import (
    digital_twin_reconstruction_loss,
    load_balance_loss,
    twin_guided_consistency_loss,
)
from .metrics import metric_dict


LOCAL_PREFIXES = (
    "personal_adapter.",
    "personal_delta_head.",
)


def is_local_parameter(
    name: str,
) -> bool:
    return name.startswith(
        LOCAL_PREFIXES
    )


def _rul_cap(
    cfg,
):
    return float(
        cfg["data"].get(
            "rul_cap",
            125.0,
        )
    )


def _zero(
    model,
):
    return next(
        model.parameters()
    ).new_tensor(0.0)


def prepare_prox_reference(
    global_state,
    device,
):
    if global_state is None:
        return None

    return {
        name: tensor.to(
            device=device,
            non_blocking=True,
        )
        for name, tensor
        in global_state.items()
        if not is_local_parameter(
            name
        )
    }


def prox_loss(
    model,
    reference,
    mu,
):
    if (
        reference is None
        or mu <= 0
    ):
        return _zero(
            model
        )

    penalty = _zero(
        model
    )

    for name, param in (
        model.named_parameters()
    ):
        if (
            name in reference
            and not is_local_parameter(
                name
            )
        ):
            penalty = (
                penalty
                + torch.sum(
                    (
                        param
                        - reference[
                            name
                        ]
                    )
                    ** 2
                )
            )

    return 0.5 * mu * penalty


def train_local_v2(
    model,
    loader,
    optimizer,
    device,
    cfg,
    global_state=None,
):
    model.train()

    use_amp = (
        device.type == "cuda"
        and cfg[
            "training"
        ].get(
            "amp",
            True,
        )
    )

    amp_dtype = (
        torch.bfloat16
        if cfg[
            "training"
        ].get(
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

    prox_reference = (
        prepare_prox_reference(
            global_state,
            device,
        )
    )

    rul_cap = _rul_cap(
        cfg
    )

    lambda_personal = float(
        cfg["training"].get(
            "lambda_personal",
            1.0,
        )
    )

    tgc_pairs = int(
        cfg["training"].get(
            "tgc_max_pairs",
            256,
        )
    )

    use_dt = bool(
        model.use_digital_twin
    )
    use_moe = bool(
        model.use_moe
    )
    use_personalization = bool(
        model.use_personalization
    )

    totals = {
        "total": 0.0,
        "shared_rul": 0.0,
        "personal_rul": 0.0,
        "dt": 0.0,
        "bal": 0.0,
        "tgc": 0.0,
        "prox": 0.0,
    }

    total_n = 0

    util_sum = None
    util_batches = 0

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
            twin_aux = twin_aux.to(
                device,
                non_blocking=True,
            )
            y = y.to(
                device,
                non_blocking=True,
            )

            target = torch.clamp(
                y / rul_cap,
                0.0,
                1.0,
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
                    use_personal_adapter=(
                        use_personalization
                    ),
                )

                shared_rul = F.mse_loss(
                    out[
                        "shared_prediction"
                    ],
                    target,
                )

                if use_personalization:
                    personal_rul = (
                        F.mse_loss(
                            out[
                                "personalized_prediction"
                            ],
                            target,
                        )
                    )

                    rul_loss = (
                        shared_rul
                        + lambda_personal
                        * personal_rul
                    ) / (
                        1.0
                        + lambda_personal
                    )
                else:
                    personal_rul = (
                        shared_rul
                    )
                    rul_loss = (
                        shared_rul
                    )

                if use_dt:
                    dt_loss = (
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
                    dt_loss = _zero(
                        model
                    )

                if use_moe:
                    bal_loss = (
                        load_balance_loss(
                            out[
                                "dense_gate"
                            ]
                        )
                    )
                else:
                    bal_loss = _zero(
                        model
                    )

                if (
                    use_dt
                    and use_moe
                    and model.router_mode
                    == "twin"
                    and cfg[
                        "training"
                    ].get(
                        "lambda_tgc",
                        0.0,
                    ) > 0
                ):
                    tgc_loss = (
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
                    tgc_loss = _zero(
                        model
                    )

                p_loss = prox_loss(
                    model,
                    prox_reference,
                    float(
                        cfg[
                            "training"
                        ].get(
                            "prox_mu",
                            0.0,
                        )
                    ),
                )

                total_loss = (
                    rul_loss
                    + float(
                        cfg[
                            "training"
                        ].get(
                            "lambda_dt",
                            0.0,
                        )
                    )
                    * dt_loss
                    + float(
                        cfg[
                            "training"
                        ].get(
                            "lambda_bal",
                            0.0,
                        )
                    )
                    * bal_loss
                    + float(
                        cfg[
                            "training"
                        ].get(
                            "lambda_tgc",
                            0.0,
                        )
                    )
                    * tgc_loss
                    + p_loss
                )

            if scaler.is_enabled():
                scaler.scale(
                    total_loss
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
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                optimizer.step()

            bs = x.size(0)
            total_n += bs

            batch_values = {
                "total": total_loss,
                "shared_rul": shared_rul,
                "personal_rul": personal_rul,
                "dt": dt_loss,
                "bal": bal_loss,
                "tgc": tgc_loss,
                "prox": p_loss,
            }

            for key, value in (
                batch_values.items()
            ):
                totals[key] += (
                    float(
                        value.detach()
                    )
                    * bs
                )

            util = (
                out[
                    "expert_utilization"
                ]
                .detach()
                .float()
                .cpu()
            )

            util_sum = (
                util
                if util_sum is None
                else util_sum + util
            )
            util_batches += 1

    result = {
        key: (
            value
            / max(
                total_n,
                1,
            )
        )
        for key, value
        in totals.items()
    }

    result[
        "expert_utilization"
    ] = (
        util_sum
        / max(
            util_batches,
            1,
        )
    )

    return result


@torch.no_grad()
def predict_v2(
    model,
    loader,
    device,
    cfg,
    collect_routing=False,
):
    model.eval()

    rul_cap = _rul_cap(
        cfg
    )

    shared_preds = []
    personal_preds = []
    targets = []
    residuals = []

    dense_gates = []
    sparse_gates = []

    use_amp = (
        device.type == "cuda"
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
        twin_aux = twin_aux.to(
            device,
            non_blocking=True,
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
                    model.use_personalization
                ),
            )

        shared = (
            out[
                "shared_prediction"
            ]
            .float()
            .cpu()
            * rul_cap
        )

        personal = (
            out[
                "personalized_prediction"
            ]
            .float()
            .cpu()
            * rul_cap
        )

        residual = (
            out[
                "residual_delta"
            ]
            .float()
            .cpu()
            * rul_cap
        )

        shared_preds.append(
            shared
        )
        personal_preds.append(
            personal
        )
        targets.append(
            y.float().cpu()
        )
        residuals.append(
            residual
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

    result = {
        "shared_pred": torch.cat(
            shared_preds
        ),
        "personal_pred": torch.cat(
            personal_preds
        ),
        "target": torch.cat(
            targets
        ),
        "residual_cycles": torch.cat(
            residuals
        ),
    }

    if collect_routing:
        result[
            "dense_gate"
        ] = torch.cat(
            dense_gates
        )
        result[
            "sparse_gate"
        ] = torch.cat(
            sparse_gates
        )

    return result


def metrics_from_prediction(
    prediction,
):
    shared_metrics = metric_dict(
        prediction[
            "shared_pred"
        ],
        prediction[
            "target"
        ],
    )

    personal_metrics = metric_dict(
        prediction[
            "personal_pred"
        ],
        prediction[
            "target"
        ],
    )

    residual_abs = torch.abs(
        prediction[
            "residual_cycles"
        ]
    )

    if residual_abs.numel():
        residual_mean = float(
            residual_abs.mean()
        )
        residual_p95 = float(
            torch.quantile(
                residual_abs,
                0.95,
            )
        )
        residual_max = float(
            residual_abs.max()
        )
    else:
        residual_mean = 0.0
        residual_p95 = 0.0
        residual_max = 0.0

    return {
        "shared": shared_metrics,
        "personalized": personal_metrics,
        "personalization_gain_rmse": (
            shared_metrics[
                "rmse"
            ]
            - personal_metrics[
                "rmse"
            ]
        ),
        "residual_abs_mean_cycles": (
            residual_mean
        ),
        "residual_abs_p95_cycles": (
            residual_p95
        ),
        "residual_abs_max_cycles": (
            residual_max
        ),
    }
