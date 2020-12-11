import os
import sys
import time

from options.test_options import TestOptions
from util.data_loader import DataLoader
from util.evaluation import ExcelEvaluate
from util.plot_handler import PlotHandler
from util.util import info, warning, get_timestamp, remove_background, normalize_with_opt

sys.path.append(os.getcwd() + "/build/cluster")  # Fix this, make it nicer
sys.path.append(os.getcwd() + "/build/map")  # Fix this, make it nicer
from map.map import Mapping

if __name__ == "__main__":
    # Set options for printing and plotting
    opt_handler = TestOptions()
    opt_handler.initialize(opt_handler.parser)
    opts = opt_handler.set_and_create()
    timestamp_run = get_timestamp()
    data_loader = DataLoader(opts, opt_handler)
    plot_handler = PlotHandler(opts, opt_handler)
    model_folder = opt_handler.set_model(opts)

    # Retrieve main parameters
    mapping_source = opts.mapping_source
    mapping_target = opts.mapping_target
    main_clusters = opts.main_clusters
    sub_clusters = opts.sub_clusters
    excel_filepath = plot_handler.plot_folder / (opts.test_set + "_" + opts.experiment_name + ".csv")
    excel = ExcelEvaluate(excel_filepath, opts.excel)

    # Look for the model
    model_search_name = "model_" + opts.method + "_main" + str(main_clusters) + "_sub" + str(sub_clusters) + "_*"
    models_files = sorted(model_folder.glob(model_search_name), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if len(models_files) < 1:
        exit("No model has been found in the " + str(model_folder) + " folder with the given parameters.")
    if -1 < opts.model_index < len(models_files):
        model_index = opts.model_index
    else:
        model_index = -1
    model = models_files[model_index]  # Get the last one (most recent)
    print("Testing will use model: " + str(model))

    # Collect the segmented image
    reference_mri = data_loader.return_model_segmented_image(model_folder, opts.method, main_clusters, sub_clusters)
    map = Mapping(data_loader, plot_handler, model_folder, main_clusters, sub_clusters, opts.method)
    map.restore_table(model)

    time_init = time.time()
    for query_filename in data_loader.all_files:
        query_friendly_filename = query_filename.name
        info("Testing with image " + query_friendly_filename + ", please make sure that you used the same "
                                                               "settings as for training.")
        # Find the query MRIs
        mris = data_loader.return_file(query_filename, query_file=True)
        truth_nonzero = None
        if 'truth' in mris:
            # Consider the truth about the tumour
            truth_mri, truth_nonzero = remove_background(mris['truth'])

            # If the truth is not there, then we don't have any tumour on this slice
            if len(truth_nonzero) == 0:
                truth_nonzero = None
                warning("The slice " + str(opts.chosen_slice) + " does not contain any tumour, "
                                                                "and thus the tumour MSE cannot be computed.")
            else:
                plot_handler.print_tumour(mris['truth'], query_friendly_filename, data_loader.mri_shape,
                                          data_loader.affine)
        info(
            "Computing mapping " + mapping_source + " to " + mapping_target + " for query " + query_friendly_filename + ".")

        mris = map.return_results_query(mris, reference_mri, opts.smoothing)
        if 'target' in mris:
            excel.evaluate(mris, query_friendly_filename, truth_nonzero, opts.smoothing)

        for label in mris.keys():
            if 'truth' not in label:
                mris[label] = normalize_with_opt(mris[label], opts.postprocess, 0.0)

        plot_handler.plot_results(mris, query_friendly_filename, opts.smoothing, data_loader.mri_shape,
                                  data_loader.affine)
    time_end = round(time.time() - time_init, 3)
    print("Time spent for testing " + str(data_loader.all_files_size) + " images " + str(time_end) + "s.")
    excel.close()
