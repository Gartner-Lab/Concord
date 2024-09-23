
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
import pandas as pd

# This method is adapted from the `scib-metrics` package
def plot_results_table(
        result_df: pd.DataFrame, 
        min_max_scale: bool = True, 
        show: bool = True, 
        save_dir: Optional[str] = None
    ) -> Table:
    """Plot the benchmarking results.

    Parameters
    ----------
    result_df
        The DataFrame containing the benchmarking results.
    min_max_scale
        Whether to min max scale the results.
    show
        Whether to show the plot.
    save_dir
        The directory to save the plot to. If `None`, the plot is not saved.
    """
    num_embeds = len(result_df.columns)
    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=matplotlib.cm.PRGn, num_stds=2.5)
    
    if min_max_scale:
        result_df = (result_df - result_df.min()) / (result_df.max() - result_df.min())
    
    # Sort by total score
    plot_df = result_df
    plot_df["Method"] = plot_df.index

    # Define column definitions
    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
    ]
    
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn(plot_df[col]),
            formatter="{:.2f}",
        )
        for col in result_df.columns
    ]
    
    # Allow to manipulate text post-hoc (in illustrator)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(result_df.columns) * 1.25, 3 + 0.3 * num_embeds))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
    
    if show:
        plt.show()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "scib_results.svg"), facecolor=ax.get_facecolor(), dpi=300)

    return tab


