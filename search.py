import os
import sys
from options.search_options import SearchOptions
from util.util import print_timestamped, info, warning, get_timestamp, mse_computation, \
    remove_background, normalize_with_opt
from util.evaluation import ExcelEvaluate
from util.data_loader import DataLoader
from util.plot_handler import PlotHandler
import time

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

    # Find the query MRI
    query_filepath = data_loader.query_folder / opts.query_filename
    query_friendly_filename = query_filepath.name
    query_mris = data_loader.return_file(query_filepath, query_file=True)
    mse_list = []
    best_filenames = []

    # Find the X MRIs with the best MSE to our source mapping
    time_init = time.time()
    for filename in data_loader.all_files:
        curr_mris = data_loader.return_file(filename)
        # Normalize image image according to the selected preprocessing
        mse = mse_computation(query_mris['source'], curr_mris['source'])
        mse_list.append(mse)
        best_filenames.append(filename)

    # Choose the X best MRIs
    chosen_filenames = [filename for _, filename in sorted(zip(mse_list, best_filenames))][:opts.n_images]
    data_loader.set_chosen_filenames(chosen_filenames)

    info("Creating cluster mapping from " + mapping_source + " to " + mapping_target + " with " + str(
        data_loader.all_files_size) + " selected training images.")

    # Collect either the segmented image, or a clustering of the first training image
    reference_mri = data_loader.return_segmented_image(data_loader.all_files[0])
    plot_handler.plot_reference(reference_mri, save_model_folder, data_loader.mri_shape, data_loader.affine,
                                opts.method, main_clusters, sub_clusters)
    print_timestamped("Retrieved data for labeled image.")
    map = Mapping(data_loader, plot_handler, save_model_folder, main_clusters, sub_clusters, opts.method)

    # Train
    map.cluster_mapping(reference_mri)

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
            plot_handler.print_tumour(query_mris['truth'], query_friendly_filename, data_loader.mri_shape,
                                      data_loader.affine)

    print_timestamped("Processing query " + query_friendly_filename + ".")

    mris = map.return_results_query(query_mris, reference_mri, opts.smoothing)
    if 'target' in mris:
        excel.evaluate(mris, query_friendly_filename, truth_nonzero, opts.smoothing)

    for label in mris.keys():
        if 'truth' not in label:
            mris[label] = normalize_with_opt(mris[label], opts.postprocess, 0.0)

    plot_handler.plot_results(mris, query_friendly_filename, opts.smoothing, data_loader.mri_shape, data_loader.affine)
    time_end = round(time.time() - time_init, 3)
    print("Time spent for searching " + str(data_loader.all_files_size) + " images " + str(time_end) + "s.")
    excel.close()
