import datetime
import time

import cv2 as cv
import numpy as np
from munkres import Munkres
from scipy.special import comb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_mutual_info_score, \
    mutual_info_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score, homogeneity_completeness_v_measure, \
    fowlkes_mallows_score


def _comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=True)


vComb = np.vectorize(_comb2)


class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_timestamped(string):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H:%M:%S')
    print(st + ": " + string)


def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H.%M.%S')


def info(string):
    print(f"{bcolor.OKBLUE}" + string + f"{bcolor.ENDC}")


def warning(string):
    print(f"{bcolor.WARNING}" + string + f"{bcolor.ENDC}")


def error(string):
    print(f"{bcolor.FAIL}" + string + f"{bcolor.ENDC}")
    exit(-1)


def return_position_indices(mri_shape):
    x = mri_shape[0]
    y = mri_shape[1]
    if len(mri_shape) == 2:
        # Retrieve all the indices of a (240, 240) image
        return np.array([[i, j] for i in range(x - 1, -1, -1) for j in range(y)])
    else:
        z = mri_shape[2]
        return np.array([[i, j, k] for i in range(x) for j in range(y) for k in range(z)])


# Handle the data
def transform(mri_scans, slice_dimension=0, twod=True, chosen_slice=None):
    # Transform the 2d matrices in column vectors
    dim_0 = slice_dimension
    dim_1 = (slice_dimension + 1) % 3
    dim_2 = (slice_dimension + 2) % 3
    mris_shape = list(mri_scans.values())[0].shape
    if twod:
        mris_shape = (mris_shape[dim_0], mris_shape[dim_1])
    transformed_matrices = {}
    if twod:
        if chosen_slice is None:
            error("Please choose a slice to create a 2D matrix.")
        for k, matrix in mri_scans.items():
            if matrix.shape[:2] != mris_shape:
                error("The MRI " + k + " does not have the same shape as the other MRIs. " +
                      "Please convert it such that the all have the same shape.")
            transformed_matrices[k] = matrix.reshape((matrix.shape[dim_0] * matrix.shape[dim_1]), matrix.shape[dim_2])[
                                      :, chosen_slice]
    else:
        for k, matrix in mri_scans.items():
            if matrix.shape != mris_shape:
                error("The MRI " + k + " does not have the same shape as the other MRIs. " +
                      "Please convert it such that the all have the same shape.")
            transformed_matrices[k] = matrix.reshape((matrix.shape[0] * matrix.shape[1] * matrix.shape[2]))

    return transformed_matrices, mris_shape


# Handle the data
def transform_single(mri, slice_dimension=0, twod=False, chosen_slice=None):
    # Transform the 2d matrices in column vectors
    dim_0 = slice_dimension
    dim_1 = (slice_dimension + 1) % 3
    dim_2 = (slice_dimension + 2) % 3
    if twod:
        mris_shape = (mri.shape[dim_0], mri.shape[dim_1])
    else:
        mris_shape = mri.shape
    if twod:
        if chosen_slice is None:
            error("Please choose a slice to create a 2D matrix.")
        reshaped_mri = mri.reshape((mri.shape[dim_0] * mri.shape[dim_1]), mri.shape[dim_2])[:, chosen_slice]
    else:
        reshaped_mri = mri.reshape((mri.shape[0] * mri.shape[1] * mri.shape[2]))

    return reshaped_mri, mris_shape


def slice_up(mri, original_shape, chosen_slice=76):
    temp = mri.reshape(original_shape)
    if len(original_shape) == 2:
        return temp
    else:
        return temp[:, :, chosen_slice]


def build_stacked_matrix(transformed_mris):
    stacked_argument = []
    def_size = list(transformed_mris.values())[0].shape
    for k, matrix in transformed_mris.items():
        if matrix.shape != def_size:
            error("All transformed matrices must have the same size.")
        stacked_argument.append(matrix)

    stacked_matrix = np.stack(stacked_argument, axis=-1)
    return stacked_matrix


def build_position_matrix(mri_scans):
    new_mris = {}
    for k, mri in mri_scans.items():
        new_mris[k] = np.array([[mri[i, j, k], (i / mri.shape[0]), (j / mri.shape[1]), (k / mri.shape[2])]
                                for i in range(mri.shape[0])
                                for j in range(mri.shape[1])
                                for k in range(mri.shape[2])], dtype=np.float_)
    return new_mris


def remove_background(transformed_mri, nonzero_indices=None):
    if nonzero_indices is None:
        nonzero_indices = np.nonzero(transformed_mri)[0]
        # We just get all possible values in a column vector
    coloured = transformed_mri[nonzero_indices]

    return coloured, nonzero_indices


def add_background(labels, desired_shape, nonzero):
    labels_with_background = np.zeros(desired_shape)
    labels_with_background[nonzero] = labels
    return labels_with_background


def adjust_labels_with_background(labels_with_background, i):
    # In this method the labels are translated from sequencing i to the others such that
    # there is colour consistency.
    m = Munkres()
    for k in labels_with_background:
        if k == i:
            continue

        contmat = contingency_matrix(labels_with_background[i], labels_with_background[k])
        minimization_matrix = contmat.max() - contmat
        munkres_tuples = m.compute(minimization_matrix)
        mapping = {}
        for t in munkres_tuples:
            # Create the mappings
            mapping[t[1]] = np.float_(t[0])
        # print(mapping)
        nonzeros = np.nonzero(labels_with_background[k])[0]
        for index in nonzeros:
            labels_with_background[k][index] = mapping[labels_with_background[k][index]]

    return labels_with_background


def map_colors(to_map, segmentation):
    m = Munkres()
    contmat = contingency_matrix(segmentation, to_map)
    print(contmat)
    nonzeros = np.nonzero(to_map)[0]
    minimization_matrix = contmat.max() - contmat
    munkres_tuples = m.compute(minimization_matrix)
    mapping = {}
    for t in munkres_tuples:
        # Create the mappings
        mapping[t[1]] = np.float_(t[0])
    for index in nonzeros:
        to_map[index] = mapping[to_map[index]]

    uniq = np.unique(to_map)
    mapping = {}
    for i in range(len(uniq)):
        mapping[uniq[i]] = np.float_(i)
    for index in nonzeros:
        to_map[index] = mapping[to_map[index]]

    return to_map


def adjust_labels_with_background_segmented(labels_with_background, segmentation):
    m = Munkres()
    for k in labels_with_background:
        contmat = contingency_matrix(segmentation, labels_with_background[k])
        nonzeros = np.nonzero(labels_with_background[k])[0]
        minimization_matrix = contmat.max() - contmat
        munkres_tuples = m.compute(minimization_matrix)
        mapping = {}
        for t in munkres_tuples:
            # Create the mappings
            mapping[t[1]] = np.float_(t[0])
        for index in nonzeros:
            labels_with_background[k][index] = mapping[labels_with_background[k][index]]

    return labels_with_background


def nonzero_union(arr1, arr2):
    return np.nonzero(arr1 * arr2)


def non_common_indices(indices1, indices2):
    return np.nonzero(np.isin(indices1, indices2, invert=True))[0], \
           np.nonzero(np.isin(indices2, indices1, invert=True))[0]


def remove_outliers(arr, scan_type):
    # Standardize
    arr = normalize_with_opt(arr, 1)
    # Don't consider the background
    partial = arr[arr > arr.min()]
    # Compute the percentiles
    q25, q75 = np.percentile(partial, 25), np.percentile(partial, 75)
    iqr = q75 - q25
    # Outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # print(lower, upper)
    # Remove outliers above
    # TODO: Is this correct? Definitely for t1 -> t2
    if "t1" in scan_type:
        arr[arr > upper] = arr.min()
    else:
        arr[arr < lower] = arr.min()
    # Normalize 0, 1
    arr = normalize_with_opt(arr, 0)

    return arr


def normalize_with_opt(arr, opt):
    # print(opt, "[", arr.min(), arr.max(), "]", end=" - ")
    if opt == 0:
        return (arr - arr.min()) / (arr.max() - arr.min())
    elif opt == 1:
        return arr - np.mean(arr[arr > 0]) / np.std(arr[arr > 0])
    # print("[", arr.min(), arr.max(), "]")
    return arr


def get_stats(truth, pred):
    contingency = contingency_matrix(truth, pred, sparse=True)
    # cm = contingency_matrix(truth, pred)
    # print(cm)
    # print(cm.sum(axis=0))
    # print(cm.sum(axis=1))
    tp_plus_fp = sum(_comb2(n_k) for n_k in np.ravel(contingency.sum(axis=0)))
    tp_plus_fn = sum(_comb2(n_c) for n_c in np.ravel(contingency.sum(axis=1)))
    tp = sum(_comb2(n_ij) for n_ij in contingency.data)
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = _comb2(truth.shape[0]) - tp - fp - fn

    return [tp, fp, fn, tn]


def crop_center(img, target_shape):
    if len(img.shape) == 3:
        x, y, z = img.shape
        cropx, cropy, cropz = target_shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        startz = z // 2 - cropz // 2
        return img[starty:starty + cropy, startx:startx + cropx, startz:startz + cropz]
    elif len(img.shape) == 2:
        x, y = img.shape
        cropx, cropy = target_shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[starty:starty + cropy, startx:startx + cropx]
    else:
        pass  # Not supported
    return img


def precision_recall(truth, pred):
    res = {}
    if len(truth) != len(pred):
        print("Something's wrong with the indices, they are not equal.")
        exit(-1)
    print_timestamped("Computing scores.")
    tp, fp, fn, tn = get_stats(truth, pred)
    # print(tp, fp, fn, tn)
    rand_index = float(tp + tn) / (tp + fp + fn + tn)
    res["RI"] = rand_index
    res["ARI"] = adjusted_rand_score(truth, pred)
    res["accuracy"] = accuracy_score(truth, pred)
    precision = float(tp) / (tp + fp)
    res["precision_cm"] = precision
    recall = float(tp) / (tp + fn)
    res["recall_cm"] = recall
    res["precision_micro"] = precision_score(truth, pred, average="micro")
    res["recall_micro"] = recall_score(truth, pred, average="micro")
    res["f1_micro"] = f1_score(truth, pred, average="micro")
    res["precision_macro"] = precision_score(truth, pred, average="macro")
    res["recall_macro"] = recall_score(truth, pred, average="macro")
    res["f1_macro"] = f1_score(truth, pred, average="macro")
    res["precision_weighted"] = precision_score(truth, pred, average="weighted")
    res["recall_weighted"] = recall_score(truth, pred, average="weighted")
    res["f1_weighted"] = f1_score(truth, pred, average="weighted")
    res["AMI"] = adjusted_mutual_info_score(truth, pred)
    res["MI"] = mutual_info_score(truth, pred)
    res["NMI"] = normalized_mutual_info_score(truth, pred)
    homo_compl_v = homogeneity_completeness_v_measure(truth, pred)
    res["homogeneity"] = homo_compl_v[0]
    res["completeness"] = homo_compl_v[1]
    res["v_measure"] = homo_compl_v[2]
    res["fowlkes"] = fowlkes_mallows_score(truth, pred)
    # print((precision * recall) ** (float(1)/2), res["fowlkes"])

    print_timestamped("Finished computing scores.")

    return res


def filter_blur(mapped, mris_shape, mode="median", k_size=3):
    r_mapped = mapped.reshape(mris_shape)
    reshape_size = 1
    for s in mris_shape:
        reshape_size *= s

    if mode == "average":
        kernel = np.ones((k_size, k_size), np.float32) / (k_size * k_size)
        dst = cv.filter2D(r_mapped, -1, kernel)
    elif mode == "median":
        dst = cv.medianBlur(np.float32(r_mapped), ksize=k_size)
    elif mode == "blur":
        dst = cv.blur(r_mapped, ksize=(k_size, k_size))
    elif mode == "gblur":
        dst = cv.GaussianBlur(r_mapped, k_size=(k_size, k_size), sigmaX=0)
    else:
        error("Smoothing mode not recognized.")

    # Now sharpen
    # dst = unsharp_mask(dst)

    return dst.reshape(reshape_size)
