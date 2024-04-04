import multiprocessing as mp
import papermill as pm

from itertools import repeat

model_ids = [
    "BluestoneLocal",
    "BluestoneUpper",
    "Coal",
    # "ElkMiddle",
    # "ElkSutton",
    # "GauleyLower",
    # "GauleySummersville",
    # "Greenbrier",
    # "Little",
    # "LowerKanawha",
    # "LowerNew",
    # "MiddleNew",
    # "UpperKanawha",
    # "UpperNew",
]

def starmap_with_kwargs(pool, fn, input_notebook, output_notebooks_location, model_ids):
    kwarg_list = [dict(model_id=model_id) for model_id in model_ids]
    args_iter = zip(repeat(input_notebook), output_notebooks_location)
    args_for_starmap = list(zip(repeat(fn), args_iter, kwarg_list))
    pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwarg):
    return fn(*args, parameters=kwarg)

def execute_notebooks(input_notebook_path, output_folder, model_ids):
    """
    Execute a given notebook for a series of model ids and save the outputs in a specified output folder.
    """
    output_notebook_list = [f"{output_folder}/{model_id}.ipynb" for model_id in model_ids]

    num_cores = mp.cpu_count()
    with mp.Pool(int(num_cores / 4)) as pool:
        starmap_with_kwargs(pool, pm.execute_notebook, input_notebook_path, output_notebook_list, model_ids)

if __name__ == "__main__":
    output_notebooks_location = "notebooks"
    input_notebook = "stac_summary.ipynb"

    execute_notebooks(input_notebook, output_notebooks_location, model_ids)
