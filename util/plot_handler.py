from util.util import info, crop_center, error, print_timestamped, normalize_with_opt
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import nibabel as nib
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

different_colors = ["#FF0000", "#008000", "#0000FF", "#FFD700",  # Red, green, blue, gold
                    "#00BFFF", "#DDA0DD", "#808080", "#800000",  # Light blue, magenta, gray, maroon
                    "#808000", "#00FF00", "#FFFF00", "#800080",  # Olive, lime, yellow, purple
                    "#008080", "#000080"]  # Teal, navy


class PlotHandler:

    def __init__(self, args, options, complementary=False):
        self.output_data_folder = options.output_data_folder
        self.phase = options.phase
        self.prefix = options.prefix
        self.mapping_source = args.mapping_source
        self.mapping_target = args.mapping_target
        self.plot_names_dict = {}
        self.plot_names_dict['source'] = self.mapping_source
        self.plot_names_dict['target'] = self.mapping_target
        self.sliced = args.sliced
        self.chosen_slice = args.chosen_slice
        self.plot_only_results = args.plot_only_results
        self.nifti_image_extension = ".nii.gz"
        self.target_shape = (200, 180)
        if self.sliced:
            self.image_extension = ".png"
        else:
            self.image_extension = self.nifti_image_extension

        # Set plot folder
        base = self.output_data_folder / args.experiment_name
        if complementary:
            specific_name = "_complementary"
        else:
            specific_name = "_" + args.method + "_main" + str(args.main_clusters) + "_sub" + str(args.sub_clusters)

        if self.phase != 1:
            specific_name += "_" + str(args.postprocess)

        if self.phase == 1:
            self.plot_folder = base / (self.prefix + specific_name)
        else:
            self.plot_folder = base / (args.test_set + specific_name)
        self.plot_folder.mkdir(parents=True, exist_ok=True)

        self.train_folder = None
        self.labels_folder = None

        info("The plots for the current run will be saved in " + str(base) + ".")
        if not args.plot_only_results and self.phase != 2:
            self.train_folder = self.plot_folder / "train"
            self.labels_folder = self.plot_folder / "labels"
            self.train_folder.mkdir(parents=True, exist_ok=True)
            self.labels_folder.mkdir(parents=True, exist_ok=True)

            print(
                "The plots of the training images for the current run will be saved in " + str(self.train_folder) + ".")
            print("The plots of the labels for the current run will be saved in " + str(self.labels_folder) + ".")

    def plot_reference(self, reference_mri, model_folder, mris_shape, affine, method, main, sub):
        plot_nifti(reference_mri,
                   model_folder / (
                           "lab_" + method + "_main" + str(main) + "_sub" + str(sub) + self.nifti_image_extension),
                   mris_shape,
                   affine)

    def plot_train(self, visuals, patient_name, mris_shape, affine):
        if not self.plot_only_results:
            for label, image in visuals.items():
                filename = self.train_folder / (patient_name + "_" + self.plot_names_dict[label] + self.image_extension)
                if self.sliced:
                    plot_image(image,
                               filename,
                               mris_shape=mris_shape,
                               plotbar=False)
                else:
                    reshaped = image.reshape(mris_shape)
                    cropped = crop_center(reshaped[:, :, self.chosen_slice], self.target_shape)
                    plot_image(cropped,
                               str(filename).split(".")[0] + ".png",
                               colormap=cm.get_cmap('gray'),
                               plotbar=False)
                    plot_nifti(image,
                               filename,
                               mris_shape,
                               affine=affine)

    def plot_results(self, visuals, patient_name, smoothing, mris_shape, affine):
        self.plot_names_dict['learned_target'] = self.mapping_target + "_learned"
        self.plot_names_dict['learned_target_smoothed'] = self.mapping_target + "_learned_" + smoothing
        for label, image in visuals.items():
            if 'truth' not in label:
                folder = self.plot_folder / patient_name
                folder.mkdir(parents=True, exist_ok=True)
                filename = folder / (patient_name + "_" + self.plot_names_dict[label] + self.image_extension)
                if self.sliced:
                    plot_image(image,
                               filename,
                               colormap=cm.get_cmap('gray'),
                               mris_shape=mris_shape,
                               plotbar=False)
                else:
                    reshaped = image.reshape(mris_shape)
                    cropped = crop_center(reshaped[:, :, self.chosen_slice], self.target_shape)
                    plot_image(cropped,
                               str(filename).split(".")[0] + ".png",
                               colormap=cm.get_cmap('gray'),
                               plotbar=False)
                    plot_nifti(image,
                               filename,
                               mris_shape,
                               affine=affine)

    def plot_shaded_labels(self, patient_name, labels1, labels2, method, main_clusters, mris_shape, affine):
        folder = (self.labels_folder / patient_name)
        folder.mkdir(parents=True, exist_ok=True)
        m1_filename = folder / (patient_name + "_" + self.mapping_source + "_labels_" + method + self.image_extension)
        m2_filename = folder / (patient_name + "_" + self.mapping_target + "_labels_" + method + self.image_extension)
        if self.sliced:
            plot_image(labels1,
                       m1_filename,
                       shaded_labels=1.0,
                       colormap=colormap_fusion(main_clusters),
                       mris_shape=mris_shape)
            plot_image(labels2,
                       m2_filename,
                       shaded_labels=1.0,
                       colormap=colormap_fusion(main_clusters),
                       mris_shape=mris_shape)
        else:
            reshaped1 = labels1.reshape(mris_shape)
            cropped1 = crop_center(reshaped1[:, :, self.chosen_slice], self.target_shape)
            plot_image(cropped1,
                       str(m1_filename).split(".")[0] + ".png",
                       shaded_labels=1.0,
                       colormap=colormap_fusion(main_clusters),
                       plotbar=False)
            reshaped2 = labels2.reshape(mris_shape)
            cropped2 = crop_center(reshaped2[:, :, self.chosen_slice], self.target_shape)
            plot_image(cropped2,
                       str(m2_filename).split(".")[0] + ".png",
                       shaded_labels=1.0,
                       colormap=colormap_fusion(main_clusters),
                       plotbar=False)
            plot_nifti(labels1,
                       m1_filename,
                       mris_shape,
                       affine=affine)
            plot_nifti(labels2,
                       m2_filename,
                       mris_shape,
                       affine=affine)

    def print_tumour(self, tumour, patient_name, mris_shape, affine):
        if not self.plot_only_results:
            folder = self.plot_folder / patient_name
            folder.mkdir(parents=True, exist_ok=True)
            filename = folder / (patient_name + "_truth_tumour" + self.image_extension)
            if self.sliced:
                plot_image(tumour,
                           filename,
                           mris_shape=mris_shape)
            else:
                reshaped = tumour.reshape(mris_shape)
                cropped = crop_center(reshaped[:, :, self.chosen_slice], self.target_shape)
                plot_image(cropped,
                           str(filename).split(".")[0] + ".png",
                           plotbar=False)
                plot_nifti(tumour,
                           filename,
                           mris_shape,
                           affine=affine)


def plot_image(image,
               filename,
               colormap=copy.copy(cm.get_cmap('viridis')),
               mris_shape=None,
               shaded_labels=None,
               one_int_bounds=False,
               plotbar=True,
               white_bg=False,
               verbose=True):
    if plotbar:
        res_size1 = 6
        res_size2 = 5
    else:
        res_size1 = res_size2 = 5
    fig = plt.figure(figsize=(res_size1, res_size2), dpi=300)
    # ax = plt.gca()
    if len(image.shape) == 1:
        if mris_shape is not None:
            image = image.reshape(mris_shape)
        else:
            error("The image cannot be reshaped and showed with imshow")
    elif len(image.shape) > 2:
        error("The image has a shape greater than 2. You might have forgotten to slice it.")
    image = np.rot90(image, k=1)
    image = np.flip(image, axis=1)
    if shaded_labels is None:
        max_lin = np.max(image)
        bounds = None
    else:
        n_clusters = int(colormap.N / 256)
        max_lin = shaded_labels
        bounds = np.linspace(0, max_lin, n_clusters + 1)
    if one_int_bounds and max_lin < 15:
        bounds = range(int(max_lin) + 1)
    min_val = np.min(image) + 0.1e-10
    if min_val > max_lin:
        min_val = max_lin
    if colormap != cm.get_cmap('gray'):
        colormap.set_under('w')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    sc = plt.imshow(image,
                    cmap=colormap,
                    vmin=min_val,
                    vmax=max_lin)
    if plotbar:
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        cax = None
        plt.colorbar(sc, cax=cax, ticks=bounds)

    plt.savefig(filename, bbox_inches='tight', transparent=not white_bg)
    if verbose:
        print_timestamped("Saved in " + str(filename))
    plt.close(fig)


def plot_nifti(image, filename, mris_shape=None, affine=None, verbose=True):
    if affine is None:
        affine = np.array([[-1., 0., 0., -0.],
                           [0., -1., 0., 239.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
    if len(image.shape) == 1:
        if mris_shape is not None:
            image = image.reshape(mris_shape)
        else:
            error("The image cannot be reshaped, please add the shape.")
    new_nifti = nib.Nifti1Image(image, affine=affine)
    nib.save(new_nifti, filename)
    if verbose:
        print_timestamped("Saved in " + str(filename))


def colormap_fusion(n_clusters):
    if n_clusters > len(different_colors):
        error("The number of clusters is greater than the available size of colours.")
    stacked_colors = []
    for i in range(n_clusters):
        colormap = shaded_color_map(different_colors[i])
        linspace_colormap = colormap(np.linspace(0.20, 1, 256))
        stacked_colors.append(linspace_colormap)

    newcolors = np.vstack(stacked_colors)
    return colors.ListedColormap(newcolors)


def full_colormap_fusion(n_clusters):
    if n_clusters > len(different_colors):
        error("The number of clusters is greater than the available size of colours.")
    return colors.ListedColormap(different_colors[:n_clusters])


def shaded_color_map(rgb_color):
    return colors.LinearSegmentedColormap.from_list("", ["white", rgb_color])
