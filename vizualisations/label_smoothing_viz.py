import torch
import torch.nn.functional as F
import pandas as pd
import altair as alt

from src.optim.reguralizer import LabelSmoothing
from vizualisations import save_chart


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )

    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=400, width=400)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            color=alt.Color(
                "target distribution:Q",
                scale=alt.Scale(scheme="viridis"),
            ),
        )
        .interactive()
    )


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[1e-9, x / d - 1e-9, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


save_chart(example_label_smoothing(), "label_smoothing_viz.html")
save_chart(penalization_visualization(), "penalization_viz.html")
