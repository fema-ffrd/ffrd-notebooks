import papermill as pm

model_ids = [
    "BluestoneLocal",
    "BluestoneUpper",
    "Coal",
    "ElkMiddle",
    "ElkSutton",
    "GauleyLower",
    "GauleySummersville",
    "Greenbrier",
    "Little",
    "LowerKanawha",
    "LowerNew",
    "MiddleNew",
    "UpperKanawha",
    "UpperNew",
]


def execute_notebooks(input_notebook_path, output_folder, model_ids):
    """
    Execute a given notebook for a series of model ids and save the outputs in a specified output folder.
    """

    for model_id in model_ids:
        print(f"Processing notebook for {model_id}")

        output_notebook = f"{output_folder}/{model_id}.ipynb"

        pm.execute_notebook(
            input_notebook_path,
            output_notebook,
            parameters={"model_id": model_id},
        )


if __name__ == "__main__":
    output_notebooks_location = "notebooks"
    input_notebook = "stac_summary.ipynb"

    execute_notebooks(input_notebook, output_notebooks_location, model_ids)
