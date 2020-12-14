from util.util import info, transform_single, warning, error, normalize_with_opt, remove_background
import kmeans1d
import nibabel as nib
import numpy as np


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
        self.preprocess = args.preprocess
        if options.segmented and self.main_clusters < 4:
            self.labeled_image = args.labeled_filename

        # Set files for training/querying
        if self.phase < 2:
            search_dir = self.train_folder
        else:
            search_dir = self.query_folder

        if self.phase == 2 and args.query_filename is not None and (search_dir / args.query_filename).exists():
            self.all_files = [search_dir / args.query_filename]
        else:
            self.all_files = sorted(search_dir.iterdir())

        self.all_files_size = len(self.all_files)

    def set_chosen_filenames(self, files):
        self.all_files = files
        self.all_files_size = len(files)

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
                "One of the shapes of the MRIs you just loaded does not correspond, this might create problems when plotting.")

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

    def return_model_segmented_image(self, model_path, method, main, sub):
        to_find = "lab_" + method + "_main" + str(main) + "_sub" + str(sub)
        files = list(model_path.glob(to_find + ".nii*"))
        img = None
        if len(files) > 0:
            nifti = nib.load(files[0])
            self.check_affine(nifti.affine)
            img = nifti.get_fdata()
            self.check_shape(img.shape)
        else:
            error("Reference image for " + to_find + " not found.")
        return np.ravel(img)

    def return_segmented_image(self, first_training):
        high_clusters = False
        reference_mri = None
        if self.labeled_image is not None:
            info("With a segmented image for label mapping.")
            reference_mri = self.load_and_check(self.labeled_image, "segmentation")
            n_labels = len(np.unique(reference_mri))
            if n_labels - 1 < self.main_clusters:
                warning("The number of main clusters selected cannot be mapped to the default segmented image. " +
                        "Selecting a training image instead.")
                high_clusters = True
            else:
                if n_labels - 1 > self.main_clusters:
                    for i in range(self.main_clusters, n_labels - 1):
                        reference_mri[np.where(reference_mri == i + 1)] = 0.0

        if self.labeled_image is None or high_clusters:
            info("The image chosen for label mapping is " + str(first_training) + ".")
            self.labeled_image = first_training
            first_image_source = self.load_and_check(first_training, self.mapping_source)
            reference_mri = np.zeros(first_image_source.shape[0])
            m1, nonzero = remove_background(first_image_source)
            # We use optimal1d
            l1, _ = kmeans1d.cluster(m1, self.main_clusters)
            l1 = np.array([(lab + 1) for lab in l1], dtype=np.float_)
            reference_mri[nonzero] = l1
        return reference_mri

    def return_file(self, filepath, query_file=False):
        imgs = {}
        # Find the source image
        source = self.load_and_check(filepath, self.mapping_source)
        if source is not None:
            source = normalize_with_opt(source, self.preprocess, 0.01)
            if self.preprocess == 1:
                source = normalize_with_opt(source, 0)
            imgs['source'] = source
        else:
            error("The folder " + str(filepath) + " does not contain any source image.")

        # Find the target image
        target = self.load_and_check(filepath, self.mapping_target)
        if target is not None:
            target = normalize_with_opt(target, self.preprocess, 0.01)
            if self.preprocess == 1:
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
