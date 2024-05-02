# -*- coding: utf-8 -*-

# Imports #####################################################################

# standard packages (remove packages when not required)
import os
import warnings
import h5py
import fsspec
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, FloatSlider, interactive, fixed
from branca.colormap import LinearColormap

from utils import get_cell_pts, get_perimeter

warnings.filterwarnings("ignore")

# assign a global font size for the plots
plt.rcParams.update({"font.size": 16})

# Functions ###################################################################


def calc_scenario_stats(df: pd.DataFrame, target_column: str):
    """ "
    Calculate the frequency, PDF, and CDF of a target column in a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe with the target column
    target_column : str
        The name of the target column in the dataframe

    Returns
    -------
    stats_df : pd.DataFrame
        A pandas dataframe with the frequency, PDF, and CDF of the target column
    """
    # filter out values of -9999 and zeros
    df = df[df[target_column] != -9999]
    df = df[df[target_column] > 0]
    # Frequency
    stats_df = (
        df.groupby(target_column)[target_column]
        .agg("count")
        .pipe(pd.DataFrame)
        .rename(columns={target_column: "frequency"})
    )
    # PDF
    stats_df["pdf"] = stats_df["frequency"] / sum(stats_df["frequency"])
    # CDF
    stats_df["cdf"] = stats_df["pdf"].cumsum() * 100
    stats_df = stats_df.reset_index()
    return stats_df


def plot_hist_cdf(
    stats_df: pd.DataFrame,
    target_column: str,
    plot_title: str,
    threshold: float,
    num_bins: int,
):
    """
    Plot a histogram and CDF of a target column in a dataframe

    Parameters
    ----------
    stats_df : pd.DataFrame
        A pandas dataframe with the frequency, PDF, and CDF of the target column
    target_column : str
        The name of the target column in the dataframe
    plot_title : str
        The title of the plot
    threshold : float
        The threshold for the histogram's x-axis
    num_bins : int
        The number of bins for the histogram
    """

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    ax = ax.flatten()
    # plot the histogram
    target_values = stats_df[target_column].values
    filtered_values = target_values[(target_values <= threshold) & (target_values >= 0)]
    weights = np.ones_like(filtered_values) / float(len(filtered_values)) * 100
    n, bins, patches = ax[0].hist(
        filtered_values,
        bins=num_bins,
        weights=weights,
        range=(0, threshold),
        density=False,
    )
    # determine the percentage of cells that have errors greater than the threshold
    if target_column == "HrsToAbsPk":
        # cells with time to peak occuring within the last hour of the simulation
        exceedance = (
            len(target_values[target_values >= threshold - 1]) / len(target_values)
        ) * 100
        plot_units = "hours"
    else:
        # cells with WSE errors greater than the threshold
        exceedance = (
            len(target_values[(target_values > threshold)]) / len(target_values)
        ) * 100
        plot_units = "ft"

    ax[0].set(
        xlabel=f"{plot_title} ({plot_units})",
        ylabel="Cell Count (%)",
        title=f"Histogram: n bins = {num_bins}",
    )
    ax[0].set_xlim(0, threshold)
    ax[0].grid()
    # add a legend
    ax[0].legend([f"{plot_title} > {threshold}: {exceedance:.2f}%"], loc="upper right")

    # plot the CDF
    cdf_df = stats_df[
        (stats_df[target_column] <= threshold) & (stats_df[target_column] >= 0)
    ]
    # Resample the data to num_bins equally spaced points
    x = np.linspace(cdf_df[target_column].min(), cdf_df[target_column].max(), num_bins)
    y = np.interp(
        x, cdf_df[target_column].sort_values(), np.linspace(0, 100, len(cdf_df))
    )
    cdf_df = pd.DataFrame({target_column: x, "cdf": y})

    ax[1].plot(cdf_df[target_column], cdf_df["cdf"], linewidth=3)
    ax[1].set(
        xlabel=f"{plot_title}",
        ylabel="P(X <= x) (%)",
        title="Non-Exceedance Probability",
    )
    ax[1].set_xlim(0, threshold)
    ax[1].set_ylim(0, 100)
    ax[1].grid()
    ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.2f}".format(x)))
    # add space between both subplots
    plt.tight_layout()
    plt.show()


def create_fmap(
    cells_gdf: gpd.GeoDataFrame,
    perimeter_gdf: gpd.GeoDataFrame,
    target_col: str,
    threshold: float,
):
    """
    Create a folium map for the QC dataset

    Parameters
    ----------
    cells_gdf : gpd.GeoDataFrame
        A geopandas dataframe with the geometry of the computational cells
    perimeter_gdf : gpd.GeoDataFrame
        A geopandas dataframe with the geometry of the domain perimeter
    target_col : str
        The name of the target column in the dataframe
    threshold : float

    Returns
    -------
    m
        A folium map object

    """
    if threshold is None:
        threshold = cells_gdf[target_col].max()
        # filter data to cells with errors greater than or equal to the threshold
        cells_gdf = cells_gdf[cells_gdf[target_col] >= threshold]
        plot_data = cells_gdf.copy()
        print(f"plotting {len(plot_data)} cells that peaked within the final hour...")
    else:
        # filter data to cells with errors greater than the threshold
        if len(cells_gdf[cells_gdf[target_col] > threshold]) == 0:
            print(
                f"No cells with errors greater than {round(threshold,2)}-ft. Select a lower threshold."
            )
            return None

        else:
            cells_gdf = cells_gdf[cells_gdf[target_col] > threshold]
            plot_data = cells_gdf.copy()
            print(
                f"plotting {len(plot_data)} cells with errors greater than {round(threshold,2)}-ft..."
            )
    # convert to epsg 4326 for folium to plot
    plot_data = plot_data.to_crs(epsg=4326)
    perimeter_gdf = perimeter_gdf.to_crs(epsg=4326)
    # Define the bounds of the data
    min_lat, min_lon = (
        plot_data.geometry.bounds.miny.min(),
        plot_data.geometry.bounds.minx.min(),
    )
    max_lat, max_lon = (
        plot_data.geometry.bounds.maxy.max(),
        plot_data.geometry.bounds.maxx.max(),
    )
    avg_lon, avg_lat = (max_lon + min_lon) / 2, (max_lat + min_lat) / 2
    # Create the folium map object
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)
    # add the perimeter to the map
    folium.GeoJson(
        perimeter_gdf,
        style_function=lambda x: {"color": "blue", "weight": 2},
    ).add_to(m)
    # Create a color scale
    min_val = plot_data[target_col].min()
    max_val = plot_data[target_col].max()
    colormap = LinearColormap(["yellow", "orange", "red"], vmin=min_val, vmax=max_val)

    # Add the GeoJson data to the map
    folium.GeoJson(
        plot_data,
        marker=folium.Circle(radius=25, fill=True, fill_opacity=0.8),
        tooltip=folium.GeoJsonTooltip(fields=["Cell", target_col]),
        popup=folium.GeoJsonPopup(fields=["Cell", target_col]),
        highlight_function=lambda x: {"fillOpacity": 0.8},
        style_function=lambda x: {
            "fillColor": colormap(x["properties"][target_col]),
            "color": colormap(x["properties"][target_col]),
            "weight": 1,
            "fillOpacity": 0.8,
        },
        zoom_on_click=True,
    ).add_to(m)

    if threshold is None:
        return m
    else:
        # add a marker for the largest error location
        max_error = plot_data[plot_data[target_col] == plot_data[target_col].max()]
        folium.Marker(
            location=[max_error.geometry.y.values[0], max_error.geometry.x.values[0]],
            popup=f"Max Error: {max_error[target_col].values[0]}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
        return m


def package_wse_error_outputs(wse_errors: np.array):
    """
    Package the Max WSE Errors from an HDF file into a geopandas dataframe
    and ipywidgets interactive plot

    Parameters
    ----------
    wse_errors : np.array
        The Max WSE Errors from an HDF file and the time of errors

    Returns
    -------
    final_df : gpd.GeoDataFrame
        A geopandas dataframe with the Max WSE Errors
    wplot : ipywidgets.interactive
        An interactive widget for plotting the histogram and CDF
    """
    # assign to a dataframe and transponse. Rows are cells
    print("Reading the data...")
    df = pd.DataFrame(wse_errors).T
    df["Cell"] = df.index
    # drop rows with zeros
    df = df.loc[(df != 0).any(axis=1)]
    # rename the columns
    df.columns = ["max_wse_error", "time_of_error", "Cell"]
    # calculcate the CDF and PDF
    print("Calculating the CDF and PDF...")
    stats_df = calc_scenario_stats(df, "max_wse_error")
    # merge the stats_df with the original dataframe based on max_wse_error
    final_df = pd.merge(df, stats_df, on="max_wse_error")
    # plot the historgram and CDF
    num_bins_slider = IntSlider(min=100, max=500, step=25, value=100)
    threshold_slider = FloatSlider(min=0.1, max=1, step=0.1, value=0.2)
    print(f"Creating the interactive plot...")
    # create an interactive widget plot
    wplot = interactive(
        plot_hist_cdf,
        stats_df=fixed(stats_df),
        target_column=fixed("max_wse_error"),
        plot_title=fixed("Max WSE Error"),
        num_bins=num_bins_slider,
        threshold=threshold_slider,
    )
    return final_df, wplot


def read_hdf_wse_errors(hdf_file_path: str, domain_name: str):
    """
    Read the Max WSE Errors from an HDF file and calculate the CDF and PDF

    Parameters
    ----------
    hdf_file_path : str
        The file path to the HDF file
    domain_name : str
        The name of the domain in the HDF file
    threshold : float
        The threshold for the histogram's x-axis
    output_directory : str
        The file path to the output directory

    Returns
    -------
    final_df : pd.DataFrame
        A pandas dataframe with the Max WSE Errors, time of error, and CDF
    """
    # open hdf from s3 uri
    if hdf_file_path.startswith("s3://"):
        with fsspec.open(hdf_file_path, mode="rb") as f:
            hdf = h5py.File(f, "r")
            hdf_path = hdf[
                f"/Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas/{domain_name}/Cell Maximum Water Surface Error"
            ]
            final_df, wplot = package_wse_error_outputs(hdf_path)
            return final_df, wplot
    # open hdf from local file path
    else:
        with h5py.File(hdf_file_path, "r") as hdf:
            hdf_path = hdf[
                f"/Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow Areas/{domain_name}/Cell Maximum Water Surface Error"
            ]
            final_df, wplot = package_wse_error_outputs(hdf_path)
            return final_df, wplot


def package_wse_ttp_outputs(
    date_time: np.array,
    wse: np.array,
):
    """
    Package the time to peak water surface elevations from an HDF file into a geopandas dataframe
    along with the unsteady state conditions and an ipywidgets interactive plot.

    Parameters
    ----------
    date_time : np.array
        The date and time of the water surface elevations read from the HDF file
    wse : np.array
        The water surface elevations read from the HDF file

    Returns
    -------
    ttp_final_df : pd.DataFrame
        A pandas dataframe with the time to peak for each cell
    uss_final_df : pd.DataFrame
        A pandas dataframe with the unsteady state conditions for each cell
    wplot : ipywidgets.interactive
        An interactive widget for plotting the histogram and CDF
    """
    print("Reading the data...")
    # create an object for datetimes
    date_time = list(date_time)
    # decode the byte strings
    date_time = [x.decode("utf-8") for x in date_time]
    # convert to a datetime object
    date_time = pd.to_datetime(date_time, format="mixed")
    # determine the start and end times
    start_time = date_time[0]
    end_time = date_time[-1]
    # conver the end time to hours
    end_time_hrs = int((end_time - start_time).total_seconds() / 3600)

    # create an dataframe for the water surface elevations
    wse_df = pd.DataFrame(wse, index=date_time).T
    wse_df["Cell"] = np.arange(0, len(wse_df))
    # convert the wse_df from wide to long format
    wse_long = wse_df.melt(id_vars="Cell", var_name="DateTimeChar", value_name="WSE")
    # drop cells with no values
    wse_long = wse_long.loc[(wse_long["WSE"] != 0)]

    print("Flagging wet cells...")
    # group the data by cell and create new columns for the min and max WSE for each group
    wet_cells = (
        wse_long.groupby("Cell")
        .agg(max_WSE=("WSE", "max"), min_WSE=("WSE", "min"))
        .reset_index()
    )
    # take the difference between the min and the max WSE
    wet_cells["diff"] = wet_cells["max_WSE"] - wet_cells["min_WSE"]
    # filter rows where the difference is not zero
    wet_cells = wet_cells[wet_cells["diff"] != 0]

    print("Calculating peaks...")
    # filter rows in the long dataframe where the cell is in the wet cells
    peak_WSE = wse_long[wse_long["Cell"].isin(wet_cells["Cell"])]
    # sort the values by cell and date
    peak_WSE = peak_WSE.sort_values(["Cell", "DateTimeChar"])
    # group the data by cell and get the max WSE
    peak_WSE["max_WSE"] = peak_WSE.groupby("Cell")["WSE"].transform("max")
    # filter rows where the WSE is equal to the max WSE
    peak_WSE = peak_WSE[peak_WSE["WSE"] == peak_WSE["max_WSE"]]
    # group the data by "Cell" and keep only the first row of each group where WSE is equal to max_WSE
    peak_WSE = peak_WSE.groupby("Cell").first().reset_index()

    print("Grouping into timesteps...")
    hrs_to_abs_peak = peak_WSE.copy()
    # calculate the difference in hours between the peak time and the start time for each cell
    hrs_to_abs_peak["HrsToAbsPk"] = (
        peak_WSE["DateTimeChar"] - start_time
    ).dt.total_seconds() / 3600

    print("Checking peaks within the final hour...")
    # filter the final time to peak dataframe to cells that peaked within the last hour of the simulation
    before_peak_df = hrs_to_abs_peak[hrs_to_abs_peak["HrsToAbsPk"] >= end_time_hrs - 1]
    at_peak_df = before_peak_df.copy()

    # add the text 'before_peak' the end of all column names
    before_peak_id = "before_peak"
    before_peak_df.columns = [
        f"{col}_{before_peak_id}"
        for col in before_peak_df.columns
        if col != f"DateTimeChar_{before_peak_id}"
    ]
    # take the 1 hour difference in WSE at each cell from its peak time
    before_peak_df[f"DateTimeChar_{before_peak_id}"] = before_peak_df[
        f"DateTimeChar_{before_peak_id}"
    ] - pd.Timedelta(hours=1)
    # lookup the WSE 1-hour before the peak time
    starting_wse_df = pd.merge(
        before_peak_df,
        wse_long,
        left_on=[f"Cell_{before_peak_id}", f"DateTimeChar_{before_peak_id}"],
        right_on=["Cell", "DateTimeChar"],
    )

    # add the text 'at_peak' the end of all column names
    at_peak_id = "at_peak"
    at_peak_df.columns = [f"{col}_{at_peak_id}" for col in at_peak_df.columns]
    # lookup the WSE at the peak time
    peak_wse_df = pd.merge(
        at_peak_df,
        wse_long,
        left_on=[f"Cell_{at_peak_id}", f"DateTimeChar_{at_peak_id}"],
        right_on=["Cell", "DateTimeChar"],
    )
    # merge the two dataframes together
    uss_final_df = pd.merge(
        starting_wse_df,
        peak_wse_df,
        left_on=f"Cell_{before_peak_id}",
        right_on=f"Cell_{at_peak_id}",
    )

    # calculate the difference in WSE between the peak time and 1-hour before the peak time
    uss_final_df["Diff"] = abs(
        uss_final_df[f"WSE_{at_peak_id}"] - uss_final_df[f"WSE_{before_peak_id}"]
    )
    # filter the dataframe to cells with 1-hour differences greater than 0.01
    uss_final_df = uss_final_df[uss_final_df["Diff"] > 0.01]

    # subset the final dataframes to the columns of interest
    uss_final_df = uss_final_df[["Diff", f"Cell_{at_peak_id}"]]
    uss_final_df.columns = ["Diff", "Cell"]
    ttp_final_df = hrs_to_abs_peak[["HrsToAbsPk", "Cell"]]

    # calculcate the CDF and PDF
    stats_df = calc_scenario_stats(ttp_final_df, "HrsToAbsPk")
    # define the min and max thresholds for the slider
    ttp_min, ttp_max = (
        ttp_final_df["HrsToAbsPk"].min(),
        ttp_final_df["HrsToAbsPk"].max(),
    )

    # plot the historgram and CDF
    num_bins_slider = IntSlider(min=100, max=500, step=25, value=100)
    threshold_slider = FloatSlider(min=ttp_min, max=ttp_max, step=5, value=ttp_max)

    wplot = interactive(
        plot_hist_cdf,
        stats_df=fixed(stats_df),
        target_column=fixed("HrsToAbsPk"),
        plot_title=fixed("Time to Peak"),
        num_bins=num_bins_slider,
        threshold=threshold_slider,
    )

    return ttp_final_df, uss_final_df, wplot


def read_hdf_wse_ttp(hdf_file_path: str, domain_name: str):
    """
    Read the water surface elevations from an HDF file and calculate the time to peak

    Parameters
    ----------
    hdf_file_path : str
        The file path to the HDF file
    domain_name : str
        The name of the domain in the HDF file

    Returns
    -------
    ttp_final_df : pd.DataFrame
        A pandas dataframe with the time to peak for each cell
    uss_final_df : pd.DataFrame
        A pandas dataframe with the unsteady state conditions for each cell
    wplot : ipywidgets.interactive
        An interactive widget for plotting the histogram and CDF
    """
    # open hdf from s3 uri
    if hdf_file_path.startswith("s3://"):
        with fsspec.open(hdf_file_path, mode="rb") as f:
            hdf = h5py.File(f, "r")
            date_time = hdf[
                f"/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp"
            ]
            wse = hdf[
                f"/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{domain_name}/Water Surface"
            ]
            ttp_final_df, uss_final_df, wplot = package_wse_ttp_outputs(date_time, wse)
            return ttp_final_df, uss_final_df, wplot
    # open hdf from local file path
    else:
        with h5py.File(hdf_file_path, "r") as hdf:
            date_time = hdf[
                f"/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp"
            ]
            wse = hdf[
                f"/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{domain_name}/Water Surface"
            ]
            ttp_final_df, uss_final_df, wplot = package_wse_ttp_outputs(date_time, wse)
            return ttp_final_df, uss_final_df, wplot


def wse_error_qc(plan_file: str, domain_name: str):
    """
    Read the Max WSE Errors from an HDF file and calculate the CDF and PDF

    Parameters
    ----------
    geometry_file : str
        The file path to the HDF geometry file
    plan_file : str
        The file path to the HDF plan file
    domain_name : str
        The name of the domain in the HDF file
    """
    # Process geometry for cells and perimeter
    cell_points_gdf = get_cell_pts(plan_file, domain_name)
    # Process Max WSE Errors and export graphics
    wse_error_df, wplot = read_hdf_wse_errors(plan_file, domain_name)
    wse_error_gdf = pd.merge(cell_points_gdf, wse_error_df, on="Cell")

    return wse_error_gdf, wplot


def plot_spatial_data(
    plan_file, domain_name: str, gdf: gpd.GeoDataFrame, qc_type: str, threshold: float
):
    """
    Plot the spatial distribution of the Max WSE errors

    Parameters
    ----------
    plan_file : str
        The file path to the HDF plan file
    domain_name : str
        The name of the domain in the HDF file
    gdf : gpd.GeoDataFrame
        A geopandas dataframe with the geometry of the computational cells the a target column for the QC data
    qc_type:
        The type of QC data to plot. One of 'max_wse_error' or 'time_to_peak'
    threshold : float
        The threshold for filtering the point data
    """
    # load the perimeter
    perimeter_gdf = get_perimeter(plan_file, domain_name)
    # create a leaflet map object
    if qc_type == "time_to_peak":
        fmap = create_fmap(gdf, perimeter_gdf, "HrsToAbsPk", threshold)
        return fmap
    elif qc_type == "max_wse_error":
        fmap = create_fmap(gdf, perimeter_gdf, "max_wse_error", threshold)
        return fmap
    else:
        print(
            'Invalid QC type. Please select one of "max_wse_error" or "time_to_peak".'
        )
        return None


def wse_ttp_qc(plan_file: str, domain_name: str):
    """
    Read the water surface elevations from an HDF file and calculate the time to peak

    Parameters
    ----------
    plan_file : str
        The file path to the HDF plan file
    domain_name : str
        The name of the domain in the HDF file
    """
    # Process geometry and export to shapefile
    cell_points_gdf = get_cell_pts(plan_file, domain_name)
    # Process WSE time to peak
    wse_time_to_peak_df, unsteady_wse_df, wplot = read_hdf_wse_ttp(
        plan_file, domain_name
    )
    wse_time_to_peak_gdf = pd.merge(cell_points_gdf, wse_time_to_peak_df, on="Cell")

    # check if steady state conditions are met within the final hour of the simulation
    if len(unsteady_wse_df) == 0:
        print("Pass: model has sufficiently peaked")
        print(
            "All cells are within 0.01 ft/hr tolerance during the final hour of the simulation"
        )
        unsteady_wse_gdf = None
        return wse_time_to_peak_gdf, unsteady_wse_gdf, wplot
    else:
        print("Fail: model has not sufficiently finished peaking.")
        print(
            f"There are {len(unsteady_wse_df)} cells with time derivatives greater than 0.01 ft/hr during the final hour of the simulation"
        )
        # create a gdf for steady state WSE
        unsteady_wse_gdf = pd.merge(cell_points_gdf, unsteady_wse_df, on="Cell")
        return wse_time_to_peak_gdf, unsteady_wse_gdf, wplot
