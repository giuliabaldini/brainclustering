import argparse
import os
import sys
from pathlib import Path

import kmeans1d
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import cm

from util import plot_handler
from util.plot_handler import full_colormap_fusion
from util.util import crop_center

original_colors = ["#FF0000", "#008000", "#0000FF", "#FFD700",  # Red, green, blue, gold
                   "#00BFFF", "#DDA0DD", "#808080", "#800000",  # Light blue, magenta, gray, maroon
                   "#808000", "#00FF00", "#FFFF00", "#800080",  # Olive, lime, yellow, purple
                   "#008080", "#000080"]  # Teal, navy

sys.path.append(os.getcwd() + "/build/cluster")  # Fix this, make it nicer
sys.path.append(os.getcwd() + "/build/map")  # Fix this, make it nicer

target_shape = (200, 180)
original_shape = (240, 240)
data_folder = Path.cwd() / "data"
output_folder = Path.cwd() / "output_data"
plot_folder = Path.cwd() / "plot"
high_size = 'HGG'
low_size = 'LGG'
test_slice = 76


def standard_view(image):
    image = np.rot90(image, k=-1)
    return np.flip(image, axis=1)


def crop_decenter(img, target_shape):
    if len(img.shape) == 3:
        pass  # Not supported
    elif len(img.shape) == 2:
        x, y = img.shape
        cropx, cropy = target_shape
        startx = (x // 2 - cropx // 2) + 8
        starty = y // 2 - cropy // 2
        return img[starty:starty + cropy, startx:startx + cropx]
    else:
        pass  # Not supported
    return img


def load_nifti(root, sequence):
    files = list(root.glob("*" + sequence + ".nii*"))
    reshaped_mri = None
    if len(files) > 0:
        nifti = nib.load(files[0])
        img = nifti.get_fdata()
        reshaped_mri = img.reshape((img.shape[0] * img.shape[1]), img.shape[2])[:, test_slice]
    return reshaped_mri


def cluster_explained():
    parser = argparse.ArgumentParser(description='Evaluation of the images in a folder.')
    parser.add_argument('--plotfolder', required=True, type=str, help='folder for plots')
    parser.add_argument('--patient', required=True, type=str, help='folder with the patient')

    args = parser.parse_args()
    plot_folder = Path.cwd() / args.plotfolder
    plot_folder.mkdir(parents=True, exist_ok=True)
    patient_file = Path.cwd() / args.patient

    n_clusters = 3
    sub_clusters = 3
    alpha_main = 0.5
    color_name = ["Red", "Blue", "Green"]
    main_colors = ["#FF0000", "#008000", "#0000FF"]  # Red, blue, green
    plot_handler.different_colors = main_colors
    cmap1 = full_colormap_fusion(n_clusters)
    cmap1.set_bad(alpha=0.0)  # set how the colormap handles 'bad' values

    plot_handler.different_colors = ["#FFFFFF", "#000000", "#808080"]
    cmap2 = full_colormap_fusion(sub_clusters)
    cmap2.set_bad(alpha=0.0)  # set how the colormap handles 'bad' values
    print(["White", "Black", "Gray"])
    for seq_name in ["t1", "t2"]:
        seq = load_nifti(patient_file, seq_name)
        seq = (seq - seq.min()) / (seq.max() - seq.min())
        nonzero_seq = np.nonzero(seq)
        clustered_seq, centers = kmeans1d.cluster(seq[nonzero_seq], n_clusters)
        print(seq_name, centers)
        mainclusters_seq = np.zeros(seq.shape)
        mainclusters_seq[nonzero_seq] = np.array([c + 1 for c in clustered_seq], dtype=np.float_)
        # Adjust view
        final_seq = standard_view(crop_decenter(seq.reshape(original_shape), target_shape))
        final_main = standard_view(crop_decenter(mainclusters_seq.reshape(original_shape), target_shape))
        final_main[final_main == 0] = np.nan

        # Normal gray brain
        fig = plt.figure(figsize=(5, 5), dpi=300)
        plt.axis('off')
        plt.imshow(final_seq, cmap=cm.get_cmap('gray'))
        plt.savefig(plot_folder / (patient_file.name + "_" + seq_name + ".png"), bbox_inches='tight', transparent=True)

        # Gray with main clusters on top
        fig = plt.figure(figsize=(5, 5), dpi=300)
        plt.axis('off')
        plt.imshow(final_seq, cmap=cm.get_cmap('gray'))
        plt.imshow(final_main, cmap=cmap1, interpolation='nearest', alpha=1.0)
        plt.savefig(plot_folder / (patient_file.name + "_" + seq_name + "_main_clusters.png"), bbox_inches='tight',
                    transparent=True)

        # Gray with main clusters with transparency
        fig = plt.figure(figsize=(5, 5), dpi=300)
        plt.axis('off')
        plt.imshow(final_seq, cmap=cm.get_cmap('gray'))
        plt.imshow(final_main, cmap=cmap1, interpolation='nearest', alpha=alpha_main)
        plt.savefig(plot_folder / (patient_file.name + "_" + seq_name + "_main_clusters_trans.png"), bbox_inches='tight',
                    transparent=True)

        subclusters_seq = np.zeros(seq.shape)
        for i in range(n_clusters):
            curr_interesting = seq[mainclusters_seq == (i + 1)]
            current_sub, centers = kmeans1d.cluster(curr_interesting, sub_clusters)
            print(i + 1, color_name[i], centers)
            subclusters_seq[mainclusters_seq == (i + 1)] = np.array([c + 1 for c in current_sub], dtype=np.float_)

        final_sub = standard_view(crop_decenter(subclusters_seq.reshape(original_shape), target_shape))
        final_sub[final_sub == 0] = np.nan
        # Sub clusters
        fig = plt.figure(figsize=(5, 5), dpi=300)
        plt.axis('off')
        plt.imshow(final_seq, cmap=cm.get_cmap('gray'))
        plt.imshow(final_sub, cmap=cmap2, interpolation='nearest', alpha=1)
        plt.imshow(final_main, cmap=cmap1, interpolation='nearest', alpha=0.5)
        plt.savefig(plot_folder / (patient_file.name + "_" + seq_name + "_sub_clusters.png"), bbox_inches='tight',
                    transparent=True)
        if seq_name == "t2":
            # For T2 we need to make the values correspond to t1
            plot_handler.different_colors = ["#FF0000", "#0000FF", "#008000"]  # Red, blue, green
            cmap1 = full_colormap_fusion(n_clusters)
            cmap1.set_bad(alpha=0.0)  # set how the colormap handles 'bad' values

            fig = plt.figure(figsize=(5, 5), dpi=300)
            plt.axis('off')
            plt.imshow(final_seq, cmap=cm.get_cmap('gray'))
            plt.imshow(final_sub, cmap=cmap2, interpolation='nearest', alpha=1)
            plt.imshow(final_main, cmap=cmap1, interpolation='nearest', alpha=0.5)
            plt.savefig(plot_folder / (patient_file.name + "_" + seq_name + "_sub_clusters_main_swapped.png"),
                        bbox_inches='tight', transparent=True)
            print("Gray, Black, White")
            plot_handler.different_colors = ["#808080", "#000000", "#FFFFFF"]
            cmap2 = full_colormap_fusion(sub_clusters)
            cmap2.set_bad(alpha=0.0)  # set how the colormap handles 'bad' values

            # Gray with main on top
            fig = plt.figure(figsize=(5, 5), dpi=300)
            plt.axis('off')
            plt.imshow(final_seq, cmap=cm.get_cmap('gray'))
            plt.imshow(final_main, cmap=cmap1, interpolation='nearest', alpha=1.0)
            plt.savefig(plot_folder / (patient_file.name + "_" + seq_name + "_main_clusters_swapped.png"),
                        bbox_inches='tight', transparent=True)

            # Gray with main on top and transparency
            fig = plt.figure(figsize=(5, 5), dpi=300)
            plt.axis('off')
            plt.imshow(final_seq, cmap=cm.get_cmap('gray'))
            plt.imshow(final_main, cmap=cmap1, interpolation='nearest', alpha=alpha_main)
            plt.savefig(plot_folder / (patient_file.name + "_" + seq_name + "_main_clusters_swapped_trans.png"),
                        bbox_inches='tight', transparent=True)

            # Sub clusters
            fig = plt.figure(figsize=(5, 5), dpi=300)
            plt.axis('off')
            plt.imshow(final_seq, cmap=cm.get_cmap('gray'))
            plt.imshow(final_sub, cmap=cmap2, interpolation='nearest', alpha=1)
            plt.imshow(final_main, cmap=cmap1, interpolation='nearest', alpha=0.5)
            plt.savefig(plot_folder / (patient_file.name + "_" + seq_name + "_sub_clusters_swapped.png"),
                        bbox_inches='tight', transparent=True)


if __name__ == "__main__":
    curr_eval_type = 0

    if curr_eval_type == 0:
        cluster_explained()
