import os
import sys
import time

import numpy as np

from options.search_options import SearchOptions
from util.data_loader import DataLoader
from util.evaluation import ExcelEvaluate
from util.plot_handler import PlotHandler
from util.util import print_timestamped, info, warning, get_timestamp, \
    remove_background, normalize_with_opt

sys.path.append(os.getcwd() + "/build/cluster")  # Fix this, make it nicer
sys.path.append(os.getcwd() + "/build/map")  # Fix this, make it nicer
from map.map import Mapping

if __name__ == "__main__":
    # Set options for printing and plotting
    opt_handler = SearchOptions()
    opt_handler.initialize(opt_handler.parser)
    opts = opt_handler.set_and_create()
    timestamp_run = get_timestamp()
    data_loader = DataLoader(opts, opt_handler)
    plot_handler = PlotHandler(opts, opt_handler)
    save_model_folder = opt_handler.set_model(opts)

    # Retrieve main parameters
    mapping_source = opts.mapping_source
    mapping_target = opts.mapping_target
    main_clusters = opts.main_clusters
    sub_clusters = opts.sub_clusters
    excel_filepath = plot_handler.plot_folder / (opts.test_set + "_" + opts.experiment_name + ".csv")
    excel = ExcelEvaluate(excel_filepath, opts.excel)

    # Iterate through the query MRIs
    time_init_total = time.time()
    for query in data_loader.query_files:
        patient_save_model_folder = save_model_folder / query.name
        patient_save_model_folder.mkdir(parents=True, exist_ok=True)
        query_mris = data_loader.return_file(query, query_file=True)
        mse_list = []

        # Find the X MRIs with the best MSE to our source mapping
        time_init = time.time()
        for filename in data_loader.train_files:
            curr_mris = data_loader.return_file(filename)
            # Normalize image image according to the selected preprocessing
            mse = np.square(np.subtract(query_mris['source'], curr_mris['source'])).mean()
            mse_list.append((mse, filename))

        # Choose the X best MRIs
        chosen_filenames = [filename for _, filename in sorted(mse_list)][:opts.n_images]
        data_loader.set_training_filenames(chosen_filenames)

        info("Creating cluster mapping from " + mapping_source + " to " + mapping_target + " with " + str(
            data_loader.train_files_size) + " selected training images.")

        # Collect either the segmented image, or a clustering of the first training image
        print_timestamped("Retrieved data for labeled image.")
        map = Mapping(data_loader, plot_handler, patient_save_model_folder, main_clusters, sub_clusters, opts.method)

        # Train
        map.cluster_mapping()

        # Check if we have a truth file to compute the MSE
        truth_nonzero = None
        if 'truth' in query_mris:
            # Consider the truth about the tumour
            truth_mri, truth_nonzero = remove_background(query_mris['truth'])

            # If the truth is not there, then we don't have any tumour on this slice
            if len(truth_nonzero) == 0:
                truth_nonzero = None
                warning("The slice " + str(opts.chosen_slice) + " does not contain any tumour, "
                                                                "and thus the tumour MSE cannot be computed.")
            else:
                plot_handler.print_tumour(query_mris['truth'], query.name, data_loader.mri_shape,
                                          data_loader.affine)

        print_timestamped("Processing query " + query.name + ".")

        mris = map.return_results_query(query_mris, opts.smoothing)
        if 'target' in mris:
            excel.evaluate(mris, query.name, truth_nonzero, opts.smoothing)

        for label in mris.keys():
            if 'truth' not in label:
                mris[label] = normalize_with_opt(mris[label], opts.postprocess)

        plot_handler.plot_results(mris, query.name, opts.smoothing, data_loader.mri_shape, data_loader.affine)
        time_end = round(time.time() - time_init, 3)
        print("Time spent for searching the current image: " + str(time_end) + "s.")
    time_end = round(time.time() - time_init_total, 3)
    print("Time spent for searching " + str(data_loader.query_files_size) + " images " + str(time_end) + "s.")
    excel.close()
