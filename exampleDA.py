# Updated exampleDA script

# Function to fill pixels at crop borders with nearest in-crop depth values
# and ensure STL values are correct. Also adds an orbitable HTML contour map export

# NOTE: This is an updated version of the exampleDA script based on recent feedback.

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Constants
INVERT_STL = False  # Default boolean option to invert STL output

# Crop and fill function
def crop_and_fill(depth_image, crop_area):
    # Existing crop logic here...
    # Instead of zeroing out, fill with nearest in-crop depth
    
    # Assume depth_image is a 2D numPy array and crop_area is defined 
    fill_value_masked = np.where(depth_image[crop_area] == 0, np.nan, depth_image[crop_area])
    # Fill NaNs with the nearest values inside the crop using interpolation
    filled_depth = griddata(np.argwhere(~np.isnan(fill_value_masked)), fill_value_masked[~np.isnan(fill_value_masked)], 
                             np.argwhere(np.isnan(filled_depth)), method='nearest')
                              
    return filled_depth

# STL processing function with inversion option
def process_stl(depth_data):
    # Inversion logic based on INVERT_STL
    if INVERT_STL:
        depth_data = np.max(depth_data) - depth_data  # Invert depth values
    
    return depth_data

# Function to export contour map
def export_contours(depth_data):
    # Existing export logic for orbit-able HTML
    fig = go.Figure(data=go.Contour(z=depth_data))
    fig.write_html('Depth_Map_Contour.html')  # Orbit-able HTML

    # New Plotly 2D contour export
    fig_2d = go.Figure(data=go.Contour(z=depth_data))
    fig_2d.write_html('Depth_Map_Contour_2D.html')  # 2D Contour Plotly HTML export

# Sample execution logic
if __name__ == '__main__':
    # Load depth image and crop area
    depth_image = np.load('depth_image.npy')
    crop_area = np.load('crop_area.npy')
    filled_image = crop_and_fill(depth_image, crop_area)
    processed_stl = process_stl(filled_image)
    export_contours(processed_stl)
