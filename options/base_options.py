import argparse
import os
from util.util import warning, error, info
from util.util import error
from pathlib import Path


# Credits: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Mapping between two MRI images.')
        self.output_data_folder = Path.cwd() / "output_data"
        self.phase = None
        self.segmented = False
        self.prefix = None

    def initialize(self, parser):
        # Folders
        parser.add_argument('--data_folder', type=str, required=True, help='folder where the data is found')
        parser.add_argument('--test_set', type=str, default="test",
                            help='the folder that contains the set of images to test.')

        # For search, have a different experiment name for every search query
        parser.add_argument('--experiment_name', type=str, default="experiment_name", help='name of the current run')

        # Parameters for 2D/3D
        parser.add_argument('--sliced', action='store_true', help='choose to run the mapping in 2D instead of 3D')
        parser.add_argument('--chosen_slice', type=int, default=76,
                            help='The slice to be considered for 2D mapping (only when sliced is true)')

        # Parameters for clustering
        parser.add_argument('--method', type=str, default="nesting", const='nesting', nargs='?',
                            choices=['kmeans', 'nesting', 'agglomerative'],
                            help='the clustering method to use, the options are kmeans, nesting or agglomerative')
        parser.add_argument('--main_clusters', type=int, default=3,
                            help='The number of main clusters, we advice to use a number between 3 and 7')
        parser.add_argument('--sub_clusters', type=int, default=201,
                            help='The number of smaller clusters, we advice to use a large number (100, 200, 500). '
                                 'The larger the number, the more the computation might take')
        parser.add_argument('--mapping_source', type=str, default="t1", const="t1", nargs='?',
                            choices=['t1', 't2', 't1ce', 'flair'],
                            help='the source sequencing for the mapping')
        parser.add_argument('--mapping_target', type=str, default="t2", const="t2", nargs='?',
                            choices=['t1', 't2', 't1ce', 'flair'],
                            help='the target sequencing for the mapping')
        parser.add_argument('--preprocess', type=int, default=0, const=0, nargs='?',
                            choices=[-1, 0, 1],
                            help='the kind of pre-processing to apply to the images. -1 means no pre-processing, '
                                 '0 means scale in range [0, 1], '
                                 '1 means normalize with unit variance and mean 0 and then scale in range [0, 1].')

        # Parameters for plotting/excel
        # TODO: Check if commenting out creates problems
        # parser.add_argument('--no_plots', action='store_true', help='choose to plot')
        parser.add_argument('--plot_only_results', action='store_true',
                            help='if plots is true, then plot only the relevant results')
        return parser

    def add_common_train_search(self, parser):
        # Labeled image
        parser.add_argument('--labeled_filename', type=str, default=None,
                            help='the file used for reference mapping, '
                                 'if none the first image of the training set will be used')
        return parser

    def add_common_test_search(self, parser):
        # Post-processing
        parser.add_argument('--smoothing', type=str, default="median", const="median", nargs='?',
                            choices=['average', 'median'],
                            help='the kind of smoothing to apply to the image after mapping')
        parser.add_argument('--excel', action='store_true',
                            help='choose to print an excel file with useful information (1) or not (0)')

        parser.add_argument('--postprocess', type=int, default=-1, const=-1, nargs='?',
                            choices=[-1, 0, 1, 2],
                            help='the kind of post-processing to apply to the images. -1 means no postprocessing, '
                                 '0 means scale in range [0, 1], '
                                 '1 means normalize with unit variance and mean 0,'
                                 'and 2 means scale to be in range [0, 1] and then normalize in range [-1, 1].')

        return parser

    def set_and_create(self):
        args = self.parser.parse_args()
        self.print_options(args)
        args.data_folder = Path(os.getcwd()) / args.data_folder

        if not args.data_folder.exists():
            error("The data folder " + str(args.data_folder) + " does not exist.")

        self.output_data_folder.mkdir(parents=True, exist_ok=True)

        if hasattr(args, 'labeled_filename'):
            self.segmented = True
            args.labeled_filename = Path(os.getcwd()) / str(args.labeled_filename)
            if not (Path.cwd() / args.labeled_filename).exists():
                warning("No segmented file has been specified, the first training image will be chosen for reference.")
                self.segmented = False

        if args.method == "kmeans" and args.sub_clusters % args.main_clusters != 0:
            # TODO: This doesn't necessarily have to be like this
            error("KMEANS: The smaller clusters cannot be equally divided over the main clusters. "
                  "Please change the number of smaller clusters to a value divisible by the number of main cluster.")

        if args.mapping_source == args.mapping_target:
            error("The mapping source and the mapping target cannot be the same.")

        return args

    def set_model(self, args):
        if self.phase < 2:
            save_output = self.output_data_folder / args.experiment_name / (self.prefix + "_model")
        else:
            save_output = self.output_data_folder / args.experiment_name / (args.model_phase + "_model")
        save_output.mkdir(parents=True, exist_ok=True)

        return save_output

    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        expr_dir = self.output_data_folder / args.experiment_name
        expr_dir.mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(self.prefix))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
