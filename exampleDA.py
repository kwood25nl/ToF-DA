def apply_crop(cropped_depth, cropped_mask):
    """ Apply crop to depth.

    This function crops the depth values according to the given mask, preserving original depth values at crop borders and outside the crop within the bbox.
    """
    # Note: cropped_depth[~cropped_mask] = 0.0 has been removed to preserve original depth values.
    return cropped_depth[cropped_mask]

STL_INVERT = True  # Default to True to match the current PLY convention.

def run():
    # ... existing code ...
    if STL_INVERT:
        stl_depth = (np.max(depth_work) - depth_work)
    else:
        stl_depth = depth_work
    build_solid_stl(stl_depth)

import plotly.graph_objects as go

def save_contour_plotly_html(depth, path, step=2):
    """ Save 2D contour plot of depth to an HTML file using plotly.
    """
    # Downsample depth data
    downsampled_depth = depth[::step, ::step]
    contour = go.Figure(data = go.Contour(z=downsampled_depth))
    contour.write_html(path)

# Call in run() to save the interactive contour plot
save_contour_plotly_html(depth, 'Depth_Map_Contour_Interactive.html')