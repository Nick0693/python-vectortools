from typing import Tuple

import numpy as np
import shapely
from simplification.cutil import simplify_coords_vwp


def vw_loop(
    x : np.ndarray, 
    y : np.ndarray, 
    max_extent : Tuple[float, float, float, float], 
    tolerance : float = 1.
    ):
    """
    Visvalingam-Whyatt algorithm loop for coordinate pairs
    
    Parameters
    ----------
        x: x-coordinates
        y: y-coordinates
        max_extent: bounds of the original raster
        tolerance (optional): tolerance of the simplification algorithm
        
    Returns
    ----------
        poly: simplified polygon
    """

    minx, miny, maxx, maxy = max_extent
    xy = np.dstack([np.asarray(x), np.asarray(y)])[0]
    coords = simplify_coords_vwp(xy, tolerance)
    coords = np.delete(coords, np.where(np.logical_or.reduce([
        coords[:,0]<minx, coords[:,0]>maxx, 
        coords[:,1]<miny, coords[:,1]>maxy
    ])), axis=0)

    poly = shapely.Polygon(coords)
    
    return poly