# -*- coding: utf-8 -*-

# Imports #####################################################################

# standard packages (remove packages when not required)
import os
import h5py
import fsspec
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import CRS
import fsspec

# Functions ###################################################################


def init_s3_keys():
    """
    Initialize the os environment variables for AWS S3
    """
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    # Set the new environment variables
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")


def get_domain_names(hdf_file_path: h5py.File):
    """
    Get all domain flow area names from the HDF file

    Parameters
    ----------
    hdf_file_path : str
        The file path to the HDF file

    Returns
    -------
    flow_areas
        A list of domain names
    """
    flow_areas = []
    omit = [
        "Attributes",
        "Cell Info",
        "Cell Points",
        "Polygon Info",
        "Polygon Parts",
        "Polygon Points",
    ]
    # open hdf from s3 uri
    if hdf_file_path.startswith("s3://"):
        with fsspec.open(hdf_file_path, mode="rb") as f:
            hdf = h5py.File(f, "r")
            hdf_path = hdf["Geometry/2D Flow Areas/"]
            for key in hdf_path:
                if key not in omit:
                    flow_areas.append(key)
                    print(key)
        if len(flow_areas) == 1:
            return flow_areas[0]
        else:
            print(
                "Multiple 2D flow areas found within HDF file. Please select one from the returned list."
            )
            return flow_areas

    else:
        # open hdf from local file path
        with h5py.File(hdf_file_path, "r") as hdf:
            hdf_path = hdf["Geometry/2D Flow Areas/"]
            for key in hdf_path:
                if key not in omit:
                    flow_areas.append(key)
                    print(key)
        if len(flow_areas) == 1:
            return flow_areas[0]
        else:
            print(
                "Multiple 2D flow areas found within HDF file. Please select one from the returned list."
            )
            return flow_areas


def decode_hdf_projection(hdf: h5py.File):
    """
    Decode the projection from the HDF file

    Parameters
    ----------
    hdf : h5py.File
        The HDF file object

    Returns
    -------
    str
        The decoded projection
    """
    proj_wkt = hdf.attrs.get("Projection")
    if proj_wkt is None:
        print("No projection found in HDF file.")
        return None
    if type(proj_wkt) == bytes or type(proj_wkt) == np.bytes_:
        proj_wkt = proj_wkt.decode("utf-8")
    return CRS.from_wkt(proj_wkt)


def get_projection(hdf_file_path: str):
    """
    Get the projection coordinate reference system from the HDF plan file

    Parameters
    ----------
    hdf_file_path : str
        The file path to the HDF file

    Returns
    -------
    hdf_proj
        The projection of the HDF file
    """
    # open hdf from s3 uri
    if hdf_file_path.startswith("s3://"):
        with fsspec.open(hdf_file_path, mode="rb") as f:
            hdf = h5py.File(f, "r")
            hdf_proj = decode_hdf_projection(hdf)
            return hdf_proj
    # open hdf from local file path
    else:
        with h5py.File(hdf_file_path, "r") as hdf:
            hdf_proj = decode_hdf_projection(hdf)
            return hdf_proj


def create_xy_gdf(coords: np.array, crs: CRS):
    """
    Create a GeoDataFrame from x and y coordinates

    Parameters
    ----------
    coords : np.array
        An array of x and y coordinates
    crs : CRS
        The coordinate reference system of the GeoDataFrame
    """
    # assign to a dataframe
    df = pd.DataFrame(coords)
    cells = df.index
    # rename the columns to x and y
    df.columns = ["x", "y"]
    # convert both columns to numeric
    df["x"] = pd.to_numeric(df["x"])
    df["y"] = pd.to_numeric(df["y"])
    # convert to a spatial geopandas dataframe
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=crs)
    gdf["Cell"] = cells
    # drop th x and y columns
    gdf = gdf.drop(columns=["x", "y"])
    return gdf


def get_cell_pts(hdf_file_path: str, domain_name: str):
    """
    Get the cell center points of the specified domain as a GeoDataFrame

    Parameters
    ----------
    hdf_file_path : str
        The file path to the HDF file
    domain_name : str
        The name of the domain in the HDF file

    Returns
    -------
    gdf : gpd.GeoDataFrame
        A geopandas dataframe with the geometry of the computational cells
    """
    # get the projection
    hdf_crs = get_projection(hdf_file_path)
    # open hdf from s3 uri
    if hdf_file_path.startswith("s3://"):
        with fsspec.open(hdf_file_path, mode="rb") as f:
            hdf = h5py.File(f, "r")
            hdf_path = hdf[
                f"Geometry/2D Flow Areas/{domain_name}/Cells Center Coordinate"
            ]
            # assign to a dataframe
            gdf = create_xy_gdf(hdf_path, hdf_crs)
            return gdf
    # open hdf from local file path
    else:
        with h5py.File(hdf_file_path, "r") as hdf:
            # navigate to the key path
            hdf_path = hdf[
                f"Geometry/2D Flow Areas/{domain_name}/Cells Center Coordinate"
            ]
            # assign to a dataframe
            gdf = create_xy_gdf(hdf_path, hdf_crs)
            return gdf

def get_perimeter(hdf_file_path: str, domain_name: str):
    """
    Get the perimeter of the specified domain as a GeoDataFrame

    Parameters
    ----------
    hdf_file_path : str
        The file path to the HDF file
    domain_name : str
        The name of the domain in the HDF file

    Returns
    -------
    gdf
        A GeoDataFrame containing the perimeter of the domain
    """
    # get the projection
    hdf_crs = get_projection(hdf_file_path)
    # open hdf from s3 uri
    if hdf_file_path.startswith("s3://"):
        with fsspec.open(hdf_file_path, mode="rb") as f:
            hdf = h5py.File(f, "r")
            perimeter_polygon = Polygon(
                hdf[f"Geometry/2D Flow Areas/{domain_name}/Perimeter"][()]
            )
            gdf = gpd.GeoDataFrame(
                {"geometry": [perimeter_polygon], "name": [domain_name]}, crs=hdf_crs
            )
            return gdf
    # open hdf from local file path
    else:
        with h5py.File(hdf_file_path, "r") as hdf:
            perimeter_polygon = Polygon(
                hdf[f"Geometry/2D Flow Areas/{domain_name}/Perimeter"][()]
            )
            gdf = gpd.GeoDataFrame(
                {"geometry": [perimeter_polygon], "name": [domain_name]}, crs=hdf_crs
            )
            return gdf

    # with h5py.File(hdf_file_path, 'r') as f:
    #     perimeter_polygon = Polygon(f[f'Geometry/2D Flow Areas/{domain_name}/Perimeter'][()])
    #     gdf = gpd.GeoDataFrame({'geometry': [perimeter_polygon], 'name': [domain_name]}, crs = hdf_crs)
    # return gdf
