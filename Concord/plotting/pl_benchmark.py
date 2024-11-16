
# Plot benchmarking results using plottable

def add_metric_row(df):
    """
    Adds a row to a multi-index DataFrame with the first element of each tuple 
    in the multi-index as values in a new 'Metric' row, and keeps only the 
    second element of the multi-index as column headers.
    
    Parameters:
        df (pd.DataFrame): A DataFrame with a multi-index in columns.
        
    Returns:
        pd.DataFrame: Modified DataFrame with a new 'Metric' row.
    """
    import pandas as pd
    df = df.copy()
    # Extract the first and second levels of the multi-index columns
    metric_labels = [col[0] for col in df.columns]
    new_columns = [col[1] for col in df.columns]
    
    # Create a new row with the metric labels
    metric_row = pd.DataFrame([metric_labels], columns=new_columns, index=["Metric"])
    
    # Update the columns of the original DataFrame to only have the second level
    df.columns = new_columns
    
    # Concatenate the metric row at the top of the original DataFrame
    result_df = pd.concat([metric_row, df], axis=0)
    
    return result_df


def plot_benchmark_table(df, pal='PRGn', cmap_method='norm', dpi=300, save_path=None, figsize=None):
    # Plot the geometry results using plotable
    from plottable import ColumnDefinition, Table
    from plottable.plots import bar
    from plottable.cmap import normed_cmap
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd

    df = df.copy()
    df = add_metric_row(df)
    num_embeds = df.shape[0]
    plot_df = df.drop("Metric", axis=0)
    plot_df['Method'] = plot_df.index

    cmap = mpl.cm.get_cmap(pal)
    if cmap_method == 'norm':
        cmap_fn = lambda col_data: normed_cmap(col_data, cmap=cmap, num_stds=2.5)
    elif cmap_method == 'minmax':
        cmap_fn = lambda col_data: mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=col_data.min(), vmax=col_data.max()),
            cmap=cmap
        ).to_rgba
    else:
        raise ValueError(f"Invalid cmap_method: {cmap_method}, choose 'norm' or 'minmax'")
    
    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
    ]

    aggr_cols = df.columns[df.loc['Metric'] == 'Aggregate score']
    stats_cols = df.columns[df.loc['Metric'] != 'Aggregate score']

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
            group=df.loc['Metric', col],
            formatter="{:.2f}",
        )
        for i, col in enumerate(stats_cols)
    ]

    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": mpl.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=df.loc['Metric', col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(aggr_cols)
    ]

    # Set figure size dynamically or use provided figsize
    if figsize is None:
        figsize = (len(df.columns) * 1.25, 3 + 0.3 * num_embeds)
    
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
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
            index_col="Method"
        ).autoset_fontcolors(colnames=plot_df.columns)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
