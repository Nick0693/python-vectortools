import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from .utils import *

logger = logging.getLogger(__name__)


def simplify(
    polygon : shapely.Polygon, 
    max_extent : Tuple[float, float, float, float],
    tolerance : float = 1., 
    how : str = 'Visvalingam-Whyatt'
):
    """
    Simplifies a polygon by removing the least impactful nodes according to
    one of two algorithms.
    
    Parameters
    ----------
        polygon: input polygon
        max_extent: bounds of the original raster
        tolerance (optional): tolerance of the simplification algorithm
        how (optional): which algorithm to use for simplification
        
    Returns
    ----------
        poly: simplified polygon
    """
    match how:
        case 'Visvalingam-Whyatt' | 'vw':
            x, y = polygon.exterior.coords.xy
            try:
                poly = vw_loop(x, y, max_extent, tolerance=tolerance)
                interiors = [x for x in polygon.interiors]
                for interior in interiors:
                    try:
                        x, y = interior.coords.xy
                        p = vw_loop(x, y, max_extent, tolerance=tolerance)
                        poly = poly.difference(p)
                    except:
                        pass
            except ValueError:
                return polygon
            
        case 'Douglas-Peucker' | 'dp':
            poly = polygon.simplify(tolerance, preserve_topology=True)
        
    return poly


def remove_small_objects(
    gdf : gpd.GeoDataFrame, 
    res : float, 
    mmu : int = 10, 
    max_iterations : int = 3
) -> gpd.GeoDataFrame:
    """
    Removes contiguous objects below a given number of pixels. Objects are merged with the neighbour
    they share the largest border with
    
    Parameters
    ----------
        gdf: geodataframe with all objects
        res: resolution of the original raster file
        mmu: the minimum mapping unit
        max_iterations: maximum number of iterations the function will attempt to remove objects.
        
    Returns
    ----------
        gdf: geodataframe with objects above the mmu
    """
    i = 0
    while not gdf[gdf.area < mmu*res].empty:
        gdf.index.name = 'oid'
        gdf = gdf.reset_index()
        below_mmu = gdf[gdf.area < mmu*res]
        gdf = gdf.drop(below_mmu.index)

        oids = []
        for index, sliver in below_mmu.iterrows():
            try:
                neighbours = gdf[gdf.touches(sliver.geometry)]
                try:
                    valid_neighbours = {}
                    for index, nb in neighbours.iterrows():
                        intersect = nb.geometry.intersection(sliver.geometry)
                        if isinstance(intersect, (shapely.LineString, shapely.MultiLineString)):
                            valid_neighbours[index] = intersect.length

                    best_nb = max(valid_neighbours, key=valid_neighbours.get)
                    oids.append(gdf.loc[best_nb, 'oid'])

                except ValueError:
                    oids.append(0)

            except:
                oids.append(0)

        below_mmu.loc[:, 'oid'] = oids
        gdf = pd.concat([gdf, below_mmu])
        gdf = (gdf
               .dissolve(by='oid')
               .explode(index_parts=False)
               .reset_index()
               .drop(columns=['oid'])
              )

        i += 1
        if i == max_iterations:
            break
            
    return gdf


def split_polygon(
    polygon : shapely.Polygon, 
    side_length : float
) -> List:
    coords = np.array(polygon.boundary.coords.xy)
    y_list = coords[1]
    x_list = coords[0]
    y1 = min(y_list)
    y2 = max(y_list)
    x1 = min(x_list)
    x2 = max(x_list)
    width = x2 - x1
    height = y2 - y1

    xcells = int(np.round(width / side_length))
    ycells = int(np.round(height / side_length))

    yi = np.linspace(y1, y2, ycells + 1)
    xi = np.linspace(x1, x2, xcells + 1)
    h_split = [
        shapely.LineString([(x, yi[0]), (x, yi[-1])]) for x in xi
    ]
    v_split = [
        shapely.LineString([(xi[0], y), (xi[-1], y)]) for y in yi
    ]
    result = polygon
    for splitter in v_split:
        result = shapely.MultiPolygon(shapely.ops.split(result, splitter))
    for splitter in h_split:
        result = shapely.MultiPolygon(shapely.ops.split(result, splitter))
    squares = list(result.geoms)

    return squares