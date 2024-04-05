import os
from typing import List

import geopandas as gpd
import pandas as pd
import pystac
import scrapbook as sb
from shapely.geometry import Point


def get_item_url(item: pystac.Item) -> str:
    """
    Parses a STAC item and retrieves the items href, removes the 'https://' protocol from it, and then appaneds it to a predefined
    URL prefix that points to the STAC browser location.

    Args:
        item (pystac.Item): The item to create the url for.

    Returns:
        str: The full URL to view the item in the STAC browser.
    """

    url_prefix = "https://radiantearth.github.io/stac-browser/#/external/"

    item_dict = item.to_dict()
    for link in item_dict["links"]:
        if link["rel"] == "self":
            href = link["href"]  # Assign the href value if 'rel' is 'self'
            break
    clean_href = href.replace("https://", "")

    return url_prefix + clean_href


def get_props(collection, model_id):
    """
    Retrieves desired properties from items for a given model id and in a given collection. Returns a GeoDataFrame with desired stats and geometry for the item.

    Args:
        collection (Collection): The collection object containing the items.
        model_id (str): The model ID used to construct item IDs. (Ex. "BluestoneLocal")

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the extracted properties and geometry for each item.
    """

    rows = []
    for i in range(1, 1001):
        item_id = f"{model_id}-{i:04}"
        try:
            item = collection.get_item(item_id)
            properties = item.properties
            item_url = get_item_url(item)

            row = {
                "computation_time_total": properties.get("results_summary:computation_time_total"),
                "error_percent": properties.get("volume_accounting:error_percent"),
                "precipitation_excess_inches": properties.get("volume_accounting:precipitation_excess_inches"),
                "Historic_Storm_Date": properties.get("FFRD:storm_historic_date"),
                "longitude": properties.get("FFRD:longitude"),
                "latitude": properties.get("FFRD:latitude"),
                "item_url": item_url,
            }
            rows.append(row)
        except:
            print(f"{item_id} not available.")

    df = pd.DataFrame(rows)
    df["geometry"] = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf.crs = "EPSG:4326"

    return gdf


def get_stats(gdf_column: gpd.GeoSeries) -> pd.DataFrame:
    """Gets stats from given pandas column"""

    stats_data = {
        "Mean": gdf_column.mean(),
        "Median": gdf_column.median(),
        "Min": gdf_column.min(),
        "Max": gdf_column.max(),
    }
    stats_df = pd.DataFrame([stats_data])

    return stats_df


def get_notebook_paths(notebooks_folder_path: str) -> List:
    """Gets path for every jupyter notebook within given folder.

    Args:
        notebooks_folder_path (str): The path to the directory containing Jupyter notebooks.

    Returns:
        List[str]: A list of jupyter notebook paths found within the specified directory.

    """
    notebook_paths = []
    for file_name in os.listdir(notebooks_folder_path):
        if file_name.endswith(".ipynb"):
            file_path = f"{notebooks_folder_path}/{file_name}"
            notebook_paths.append(file_path)
    return notebook_paths


def process_notebook_results(notebook_paths: List[str]) -> pd.DataFrame:
    """Loops through jupyter notebooks and uses scrapbook to gather stats from each one, converting the stats to a dataframe.

    Args:
        notebook_paths (List[str]): A list of file paths to Jupyter notebooks.

    Returns:
        pd.DataFrame: A DataFrame containing aggregated statistical data extracted from each notebook.

    """
    all_data_df = pd.DataFrame()
    for notebook_path in notebook_paths:

        nb = sb.read_notebook(notebook_path)
        scraps = nb.scraps
        stats_df = pd.DataFrame([scraps["summary_stats"].data])

        all_data_df = pd.concat([all_data_df, stats_df], ignore_index=True)

    all_data_df = all_data_df[
        ["model_id", "compute_time_avg", "pct_error_avg", "precip_excess_avg", "precip_excess_max", "precip_excess_min"]
    ]
    all_data_df.rename(
        columns={
            "model_id": "Model ID",
            "compute_time_avg": "Avg Compute Time (Min)",
            "pct_error_avg": "Avg Volume Error (%)",
            "precip_excess_avg": "Avg Precip Excess (in)",
            "precip_excess_max": "Max Precip Excess (in)",
            "precip_excess_min": "Min Precip Excess (in)",
        },
        inplace=True,
    )

    df_rounded = all_data_df.round(2)
    return df_rounded


def convert_paths_to_html_links(data_frame: pd.DataFrame, notebook_paths: List[str]) -> pd.DataFrame:
    """Converts a list of Jupyter notebook paths to HTML file paths and adds them as hyperlinks
    in a new column of the provided DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame which HTML hyperlinks will be added to.
        notebook_paths (List[str]): A list of file paths to Jupyter notebooks, which will be
                                    converted to their HTML versions.

    Returns:
        pd.DataFrame: The modified DataFrame with an additional column "HTML_Link" containing
                        HTML hyperlinks to the notebooks.

    """

    html_paths = [path.replace(".ipynb", ".html") for path in notebook_paths]
    data_frame["HTML_Link"] = html_paths
    data_frame["HTML_Link"] = data_frame["HTML_Link"].apply(lambda x: f"<a href='{x}'>{x}</a>")
    return data_frame
