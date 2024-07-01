import os

def save_chart(
    chart, filename, output_dir="output/figures", format="html", scale_factor=2.0
):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    chart.save(
        filepath,
        format=format,
        scale_factor=scale_factor,
        embed_options={"renderer": "svg"},
    )
