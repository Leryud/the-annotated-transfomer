import torch
import pandas as pd
import altair as alt
from torch.optim.lr_scheduler import LambdaLR

from src.optim.optimizer import rate
from vizualisations import save_chart


def example_lr_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]
    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    for idc, example in enumerate(opts):
        optimizer = torch.optim.Adam(
            dummy_model.parameters(),
            lr=1,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )

        tmp = []
        # 20K dummy train steps, save the LR at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enabling Altair to handle more that 5K rows
    alt.data_transformers.disable_max_rows()
    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "warmup": ["512:4000", "512:8000", "256:4000"][warmup_idx],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="warmup:N")
        .interactive()
    )


save_chart(example_lr_schedule(), "lr_schedule.html")
