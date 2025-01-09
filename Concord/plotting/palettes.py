# Define palettes for plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


NUMERIC_PALETTES = {
    "BlueGreenRed": ["midnightblue", "dodgerblue", "seagreen", "#00C000", "#EEC900", "#FF7F00", "#FF0000"],
    "RdOgYl": ["#D9D9D9", "red", "orange", "yellow"],
    "grey&red": ["grey", "#b2182b"],
    "blue_green_gold": ["#D9D9D9", "blue", "green", "#FFD200", "gold"],
    "black_red_gold": ["#D9D9D9", "black", "red", "#FFD200"],
    "black_red": ["#D9D9D9", "black", "red"],
    "red_yellow": ["#D9D9D9", "red", "yellow"],
    "black_yellow": ["#D9D9D9", "black", "yellow"],
    "black_yellow_gold": ["#D9D9D9", "black", "yellow", "gold"],
}


def extend_palette(base_colors, num_colors):
    """
    Extends the palette to the required number of colors using linear interpolation.
    """
    base_colors_rgb = [mcolors.to_rgb(color) for color in base_colors]
    extended_colors = []

    for i in range(num_colors):
        # Calculate interpolation factor
        scale = i / max(1, num_colors - 1)
        # Determine positions in the base palette to blend between
        index = scale * (len(base_colors_rgb) - 1)
        lower_idx = int(np.floor(index))
        upper_idx = min(int(np.ceil(index)), len(base_colors_rgb) - 1)
        fraction = index - lower_idx

        # Linear interpolation between two colors
        color = [
            (1 - fraction) * base_colors_rgb[lower_idx][channel] + fraction * base_colors_rgb[upper_idx][channel]
            for channel in range(3)
        ]
        extended_colors.append(color)

    return [mcolors.rgb2hex(color) for color in extended_colors]

def get_factor_color(labels, pal='Set1', permute=True, seed=42):
    from natsort import natsorted
    # Convert labels to strings and replace 'nan' with 'NaN'
    labels = pd.Series(labels).astype(str)
    labels[labels == 'nan'] = 'NaN'

    unique_labels = labels.unique()
    # Sort unique labels to ensure consistent color assignment
    unique_labels = natsorted(unique_labels)
    
    has_nan = 'NaN' in unique_labels

    if has_nan:
        unique_labels_non_nan = [label for label in unique_labels if label != 'NaN']
    else:
        unique_labels_non_nan = unique_labels

    num_colors = len(unique_labels_non_nan)
    light_grey = '#d3d3d3'  # Define light grey color

    # Generate colors for non-NaN labels, excluding light grey
    if pal in NUMERIC_PALETTES:
        colors = NUMERIC_PALETTES[pal]
        colors = [color for color in colors if color.lower() != light_grey]  # Remove light grey if present
        if len(colors) < num_colors:
            colors = extend_palette(colors, num_colors)  # Extend the palette to match the number of labels
        else:
            colors = colors[:num_colors]
    else:
        try:
            base_palette = sns.color_palette(pal)
            max_palette_colors = len(base_palette)
            colors = sns.color_palette(pal, min(num_colors, max_palette_colors))
            colors_hex = [mcolors.rgb2hex(color) for color in colors]
            colors_hex = [color for color in colors_hex if color.lower() != light_grey]

            if num_colors > len(colors_hex):
                # Extend the palette if more colors are needed
                colors = extend_palette(colors_hex, num_colors)
            else:
                colors = colors_hex
        except ValueError:
            # Default to 'Set1' if palette not found
            colors = sns.color_palette('Set1', min(num_colors, len(sns.color_palette('Set1'))))
            colors_hex = [mcolors.rgb2hex(color) for color in colors]
            if num_colors > len(colors_hex):
                colors = extend_palette(colors_hex, num_colors)
            else:
                colors = colors_hex

    if permute:
        np.random.seed(seed)
        np.random.shuffle(colors)

    # Map colors to non-NaN labels
    color_map = dict(zip(unique_labels_non_nan, colors))

    # Assign light grey to 'NaN' label
    if has_nan:
        color_map['NaN'] = light_grey

    return color_map



def get_numeric_color(pal='RdYlBu'):
    if pal in NUMERIC_PALETTES:
        colors = NUMERIC_PALETTES[pal]
        cmap = mcolors.LinearSegmentedColormap.from_list(pal, colors)
    elif pal in plt.colormaps():
        cmap = plt.get_cmap(pal)
    else:
        cmap = sns.color_palette(pal, as_cmap=True)
    return cmap




def get_color_mapping(adata, col, pal, seed=42):
    import scipy.sparse as sp
    """Generate color map or palette based on column data type in adata.obs or adata.var."""
    if col is None:
        return None, None, None  # No coloring

    # Check if `col` is in adata.obs
    if col in adata.obs:
        data_col = adata.obs[col]
    # Check if `col` is a gene in adata.var_names
    elif col in adata.var_names:
        # Extract gene expression values
        data_col = adata[:, col].X
        if sp.issparse(data_col):  # Convert sparse matrix to dense array if necessary
            data_col = data_col.toarray().flatten()
    else:
        raise ValueError(f"'{col}' is neither a column in adata.obs nor a gene in adata.var_names.")
    
    # Get the current palette
    current_pal = pal.get(col, None) if isinstance(pal, dict) else pal

    # If `current_pal` is already a dict, just return it
    if isinstance(current_pal, dict):
        if pd.api.types.is_numeric_dtype(data_col):
            raise ValueError("A numeric column cannot use a dict palette. Please provide a colormap instead.")
        data_col = pd.Series(data_col).astype(str).replace('nan', 'NaN')
        adata.obs[col] = data_col
        return data_col, None, current_pal

    # Handle numeric data
    if pd.api.types.is_numeric_dtype(data_col):
        if current_pal is None:
            current_pal = 'viridis'  # Default colormap for numeric data
        cmap = get_numeric_color(current_pal)
        palette = None
    # Handle categorical data
    else:
        data_col = data_col.copy().astype(str)
        data_col[data_col == 'nan'] = 'NaN'
        adata.obs[col] = data_col
        if current_pal is None:
            current_pal = 'Set1'  # Default palette for categorical data
        color_map = get_factor_color(data_col, current_pal, seed=seed)
        categories = data_col.astype('category').cat.categories
        # Create a dictionary palette mapping categories to colors
        palette = {cat: color_map[cat] for cat in categories}
        cmap = None

    return data_col, cmap, palette


