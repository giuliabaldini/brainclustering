import os
import sys
import time

from options.train_options import TrainOptions
from util.data_loader import DataLoader
from util.plot_handler import PlotHandler
from util.util import print_timestamped, info

sys.path.append(os.getcwd() + "/build/cluster")  # Fix this, make it nicer
sys.path.append(os.getcwd() + "/build/map")  # Fix this, make it nicer
from map.map import Mapping

if __name__ == "__main__":
    # Set options for printing and plotting
    opt_handler = TrainOptions()
    opt_handler.initialize(opt_handler.parser)
    opts = opt_handler.set_and_create()
    data_loader = DataLoader(opts, opt_handler)
    plot_handler = PlotHandler(opts, opt_handler)
    save_model_folder = opt_handler.set_model(opts)

    # Save important values
    mapping_source = opts.mapping_source
    mapping_target = opts.mapping_target
    main_clusters = opts.main_clusters
    sub_clusters = opts.sub_clusters

    info("Creating cluster mapping from " + mapping_source + " to " + mapping_target +
         " with " + str(data_loader.train_files_size) + " training images.")

    time_init = time.time()
    print_timestamped("Retrieved data for labeled image.")
    map = Mapping(data_loader, plot_handler, save_model_folder, main_clusters, sub_clusters, opts.method)

    # Train
    table = map.cluster_mapping()
    time_end = round(time.time() - time_init, 3)
    print("Time spent for training " + str(data_loader.train_files_size) + " images " + str(time_end) + "s.")
