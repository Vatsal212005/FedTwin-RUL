import torch
import torch.nn.functional as F

from .losses import (
    digital_twin_reconstruction_loss,
)
from .metrics import metric_dict


def _rul_cap(
    cfg,
):
    return float(
        cfg[
            "data"
        ].get(
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
    }


def fedprox_penalty(
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
        if name in reference:
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

    return (
        0.5
        * float(mu)
        * penalty
    )


def train_local_tcm(
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

    total_loss_sum = 0.0
    rul_loss_sum = 0.0
    dt_loss_sum = 0.0
    prox_loss_sum = 0.0
    total_n = 0

    for _ in range(
        cfg[
            "training"
        ][
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
                )

                rul_loss = F.mse_loss(
                    out[
                        "prediction"
                    ],
                    target,
                )

                if (
                    model.use_dt_reconstruction
                    and cfg[
                        "training"
                    ].get(
                        "lambda_dt",
                        0.0,
                    ) > 0
                ):
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

                prox_loss = (
                    fedprox_penalty(
                        model,
                        prox_reference,
                        cfg[
                            "training"
                        ].get(
                            "prox_mu",
                            0.0,
                        ),
                    )
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
                    + prox_loss
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

            bs = x.size(
                0
            )

            total_n += bs

            total_loss_sum += (
                float(
                    total_loss.detach()
                )
                * bs
            )

            rul_loss_sum += (
                float(
                    rul_loss.detach()
                )
                * bs
            )

            dt_loss_sum += (
                float(
                    dt_loss.detach()
                )
                * bs
            )

            prox_loss_sum += (
                float(
                    prox_loss.detach()
                )
                * bs
            )

    denom = max(
        total_n,
        1,
    )

    return {
        "loss": (
            total_loss_sum
            / denom
        ),
        "rul_loss": (
            rul_loss_sum
            / denom
        ),
        "dt_loss": (
            dt_loss_sum
            / denom
        ),
        "prox_loss": (
            prox_loss_sum
            / denom
        ),
    }


@torch.no_grad()
def predict_tcm(
    model,
    loader,
    device,
    cfg,
    collect_modulation=False,
):
    model.eval()

    use_amp = (
        device.type == "cuda"
    )

    rul_cap = _rul_cap(
        cfg
    )

    preds = []
    targets = []

    gamma_shift = []
    beta_magnitude = []
    twin_norm = []

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
            )

        pred = (
            out[
                "prediction"
            ]
            .float()
            .cpu()
            * rul_cap
        )

        preds.append(
            pred
        )

        targets.append(
            y.float().cpu()
        )

        if collect_modulation:
            gamma = (
                out[
                    "gamma"
                ]
                .float()
                .cpu()
            )

            beta = (
                out[
                    "beta"
                ]
                .float()
                .cpu()
            )

            twin = (
                out[
                    "twin_state"
                ]
                .float()
                .cpu()
            )

            gamma_shift.append(
                torch.mean(
                    torch.abs(
                        gamma
                        - 1.0
                    ),
                    dim=-1,
                )
            )

            beta_magnitude.append(
                torch.mean(
                    torch.abs(
                        beta
                    ),
                    dim=-1,
                )
            )

            twin_norm.append(
                torch.linalg.vector_norm(
                    twin,
                    dim=-1,
                )
            )

    result = {
        "pred": torch.cat(
            preds
        ),
        "target": torch.cat(
            targets
        ),
    }

    if collect_modulation:
        result.update({
            "gamma_abs_shift": (
                torch.cat(
                    gamma_shift
                )
            ),
            "beta_abs_mean": (
                torch.cat(
                    beta_magnitude
                )
            ),
            "twin_state_norm": (
                torch.cat(
                    twin_norm
                )
            ),
        })

    return result


def evaluate_tcm(
    model,
    loader,
    device,
    cfg,
):
    pred = predict_tcm(
        model,
        loader,
        device,
        cfg,
        collect_modulation=False,
    )

    return metric_dict(
        pred["pred"],
        pred["target"],
    )
