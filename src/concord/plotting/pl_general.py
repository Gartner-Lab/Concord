

def plot_bar(
    data, key, color='#1f77b4', title=None, 
    order=True, ascending=True, log_scale=False,
    barh=True, legend=False,
    save_path=None, figsize=(4, 3), dpi=300, 
    xlabel_fontsize=8, ylabel_fontsize=8, 
    legend_fontsize=8,
    tick_fontsize=7, title_fontsize=9, bar_width=0.8
):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Sort the data based on values in the specified column
    if order:
        data = data.sort_values(by=key, ascending=ascending)
        
    # Plot as horizontal or vertical bar chart
    if barh:
        data.plot(
            kind='barh', y=key, ax=ax, color=color, 
            width=bar_width
        )
        if log_scale:
            ax.set_xscale('log')
    else:
        data.plot(
            kind='bar', y=key, ax=ax, color=color, 
            width=bar_width
        )
        if log_scale:
            ax.set_yscale('log')
    
    # Set title and font sizes
    if title is None:
        title = key
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(key, fontsize=xlabel_fontsize)
    ax.set_ylabel(data.index.name if data.index.name else '', fontsize=ylabel_fontsize)
    
    # Set tick font sizes
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    
    # Legend settings
    if legend:
        ax.legend(
            title=None,
            loc='center left', 
            bbox_to_anchor=(1, 0.5),  # Position to the right, centered vertically
            handletextpad=0.2,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize
        )
    else:
        ax.get_legend().remove()
    
    # Save the plot
    if save_path:
        fig.savefig(save_path, dpi=dpi)
        
    plt.show()
    plt.close(fig)
