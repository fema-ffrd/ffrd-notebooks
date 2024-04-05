# ffrd-notebooks

# Purpose
To allow for easy access to FFRD data that is located within SpatioTemporal Asset Catalogs (STAC).

# Summary:
Jupyter Notebooks for interacting with and displaying FFRD data from STAC. Creates a jupyter notebook for a list of given model ids (ex. BluestoneLocal) with an interactive map that shows storm
centers for all events simulated and their associated date. Also includes model statistics and plots for total compute time, volume error, and excess precipitation.

# How-to:
1. In stac_summary.ipynb, adjust the api_url and storm_items_collection_name as needed.
2. Adjust the model_ids list if needed in run_notebooks.py
3. Run run_notebooks.py to create a notebook for each model_id in the list of model_ids from run_notebooks.py. Adjust the output_notebooks_location if desired.
4. Run notebooks_to_html.py to convert all the jupyter notebooks to html, ensuring that notebooks_dir is correctly set
5. Run notebook_results.ipynb to produce a summary table of each notebook, with a link to the notebooks html file
6. If desired, convert the notebook_results notebook to an html. Can use command line args similar to whats in notebooks_to_html.py ex. jupyter nbconvert (file_path) --to html_embed --no-input