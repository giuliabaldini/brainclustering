import nibabel as nib
import numpy as np

from util.util import transform_single, warning, error, remove_outliers, normalize_with_opt


class DataLoader:
    def __init__(self, args, options):
        self.train_folder = args.data_folder / "train"
        self.query_folder = args.data_folder / args.test_set
        self.phase = options.phase
        self.prefix = options.prefix
        self.mapping_source = args.mapping_source
        self.mapping_target = args.mapping_target
        self.main_clusters = args.main_clusters
        self.sliced = args.sliced
        self.chosen_slice = args.chosen_slice
        self.affine = None
        self.mri_shape = None
        self.labeled_image = None
        if options.segmented and self.main_clusters < 4:
            self.labeled_image = args.labeled_filename

        # Set files for training/querying
        self.train_files = sorted(self.train_folder.iterdir())
        self.query_files = sorted(self.query_folder.iterdir())

        self.train_files_size = len(self.train_files)
        self.query_files_size = len(self.query_files)

    def set_training_filenames(self, files):
        self.train_files = files
        self.train_files_size = len(files)

    def check_affine(self, new_affine):
        if self.affine is None:
            self.affine = new_affine
        elif not np.array_equal(self.affine, new_affine):
            warning("One of the affines you just loaded does not correspond, this might create problems when plotting.")

    def check_shape(self, new_shape):
        if self.mri_shape is None:
            self.mri_shape = new_shape
        elif self.mri_shape != new_shape:
            warning(
                "One of the shapes of the MRIs you just loaded does not correspond, "
                "this might create problems when plotting.")

    def load_and_check(self, root, name):
        files = list(root.glob("*" + name + ".nii*"))
        t_img = None
        if len(files) > 0:
            nifti = nib.load(files[0])
            self.check_affine(nifti.affine)
            img = nifti.get_fdata()
            t_img, mri_shape = transform_single(img, twod=self.sliced, chosen_slice=self.chosen_slice)
            self.check_shape(mri_shape)
        return t_img

    def return_file(self, filepath, query_file=False):
        imgs = {}
        # Find the source image
        source = self.load_and_check(filepath, self.mapping_source)
        if source is not None:
            # TODO: Remove outliers works for T1, but doesn't for T2
            source = remove_outliers(source, self.mapping_source)
            imgs['source'] = source
        else:
            error("The folder " + str(filepath) + " does not contain any source image.")

        # Find the target image
        target = self.load_and_check(filepath, self.mapping_target)
        if target is not None:
            # TODO: Find a way to make the T2 clusters always compatible
            # target = remove_outliers(target, self.mapping_target)
            target = normalize_with_opt(target, 0)
            imgs['target'] = target

        # If we are in search or train, we need both target and source
        if not query_file and 'target' not in imgs:
            error("The folder " + str(filepath) + " does not contain any target image.")

        # We need the truth only in test or for one search image
        if query_file and 'target' in imgs:
            truth = self.load_and_check(filepath, "truth")
            if truth is not None:
                imgs['truth'] = truth

        return imgs
