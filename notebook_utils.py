import os

import geopandas as gpd
import pandas as pd
import scrapbook as sb
from shapely.geometry import Point


def get_props(collection, model_id):
    """Retrieves desired properties from items for a given model id and in a given collection. Returns a GeoDataFrame with desired stats and geometry for the item."""

    rows = []
    for i in range(1, 1001):
        item_id = f"{model_id}-{i:04}"
        try:
            item = collection.get_item(item_id)
            properties = item.properties

            row = {
                "computation_time_total": properties.get("results_summary:computation_time_total"),
                "error_percent": properties.get("volume_accounting:error_percent"),
                "precipitation_excess_inches": properties.get("volume_accounting:precipitation_excess_inches"),
                "Historic_Storm_Date": properties.get("FFRD:storm_historic_date"),
                "longitude": properties.get("FFRD:longitude"),
                "latitude": properties.get("FFRD:latitude"),
            }
            rows.append(row)
        except:
            print(f"{item_id} not available.")

    df = pd.DataFrame(rows)
    df["geometry"] = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf.crs = "EPSG:4326"

    return gdf


def get_stats(gdf_column):
    """Gets stats from given pandas column"""

    stats_data = {
        "Mean": gdf_column.mean(),
        "Median": gdf_column.median(),
        "Min": gdf_column.min(),
        "Max": gdf_column.max(),
    }
    stats_df = pd.DataFrame([stats_data])

    return stats_df


def get_notebook_paths(notebooks_folder_path):
    """Gets path for every jupyter notebook within given folder."""
    notebook_paths = []
    for file_name in os.listdir(notebooks_folder_path):
        if file_name.endswith(".ipynb"):
            file_path = notebooks_folder_path + "/" + file_name
            notebook_paths.append(file_path)
    return notebook_paths


def process_notebook_results(notebook_paths):
    """Loops through jupyter notebooks and uses scrapbook to gather stats from each one, converting the stats to a dataframe"""
    all_data_df = pd.DataFrame()
    for notebook_path in notebook_paths:

        nb = sb.read_notebook(notebook_path)
        scraps = nb.scraps
        scraps["summary_stats"].data
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


def convert_paths_to_html_links(data_frame, notebook_paths):
    """Add HTML hyperlinks to dataframe"""

    html_paths = [path.replace(".ipynb", ".html") for path in notebook_paths]
    data_frame["HTML_Link"] = html_paths
    data_frame["HTML_Link"] = data_frame["HTML_Link"].apply(lambda x: f"<a href='{x}'>{x}</a>")
    return data_frame
