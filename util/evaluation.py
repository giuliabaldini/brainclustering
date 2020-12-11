import numpy as np
import time
from util.util import error, print_timestamped, common_nonzero, normalize_with_opt


class ExcelEvaluate:
    def __init__(self, filepath, excel=False):
        self.excel_filename = None
        self.excel = excel
        if self.excel:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.excel_filename = filepath

            ff = open(self.excel_filename, "w")
            self.ff = ff
            init_rows = [
                "query_filename",
                "filter",
                "MSE",
                "relMSE",
                "TumourMSE",
                "sMAPE",
                "SMSE",
                "scaledMSE",
                "scaledrelMSE",
                "scaledTumourMSE",
                "scaledsMAPE",
                "scaledSMSE",
            ]
            for i, n in enumerate(init_rows):
                self.ff.write(n)
                if i < len(init_rows) - 1:
                    self.ff.write(",")
                else:
                    self.ff.write("\n")

    def print_to_excel(self, data):
        for i, d in enumerate(data):
            self.ff.write(str(d))
            if i < len(data) - 1:
                self.ff.write(",")
            else:
                self.ff.write("\n")

    def evaluate(self, mri_dict, query_name, truth_nonzero, smoothing):
        mse, relmse, tumour, smape, smse = evaluate_result(mri_dict['target'],
                                                           mri_dict['learned_target'],
                                                           tumour_indices=truth_nonzero,
                                                           mris_shape=None)
        s_mse, s_relmse, s_tumour, s_smape, s_smse = evaluate_result(mri_dict['target'],
                                                                     mri_dict['learned_target_smoothed'],
                                                                     tumour_indices=truth_nonzero,
                                                                     mris_shape=None)
        print_timestamped("Computing MSE on the scaled data")
        r_real = normalize_with_opt(mri_dict['target'], 0)
        r_predicted = normalize_with_opt(mri_dict['learned_target'], 0)
        r_predicted_smoothed = normalize_with_opt(mri_dict['learned_target_smoothed'], 0)
        scaledmse, scaledrelmse, scaledtumour, scaledsmape, scaledsmse = evaluate_result(r_real,
                                                                                         r_predicted,
                                                                                         tumour_indices=truth_nonzero,
                                                                                         mris_shape=None)
        scaleds_mse, scaleds_relmse, scaleds_tumour, scaleds_smape, scaleds_smse = evaluate_result(r_real,
                                                                                                   r_predicted_smoothed,
                                                                                                   tumour_indices=truth_nonzero,
                                                                                                   mris_shape=None)
        if smoothing == "median":
            smoothing = 0
        elif smoothing == "average":
            smoothing = 1
        if self.excel:
            self.print_to_excel([query_name, -1,
                                 mse, relmse, tumour, smape, smse,
                                 scaledmse, scaledrelmse, scaledtumour, scaledsmape, scaledsmse,
                                 ])
            self.print_to_excel([query_name, smoothing,
                                 s_mse, s_relmse, s_tumour, s_smape, s_smse,
                                 scaleds_mse, scaleds_relmse, scaleds_tumour, scaleds_smape, scaleds_smse
                                 ])

    def close(self):
        if self.excel:
            self.ff.close()
            print_timestamped("Saved in " + str(self.excel_filename))


def square_index(curr, shape, radius):
    indices = []
    d_index = np.unravel_index(curr, shape)
    x = d_index[0]
    y = d_index[1]
    if len(shape) == 2:
        for i in range(x - radius, x + radius + 1):
            if i < 0 or i >= shape[0]:
                continue
            for j in range(y - radius, y + radius + 1):
                if j < 0 or j >= shape[1]:
                    continue
                indices.append(i * shape[1] + j)
    else:
        # TODO: Check this to make sure it is correct
        #   either way, this is not really feasible
        z = d_index[2]
        for i in range(x - radius, x + radius + 1):
            if i < 0 or i >= shape[0]:
                continue
            for j in range(y - radius, y + radius + 1):
                if j < 0 or j >= shape[1]:
                    continue
                for k in range(z - radius, z + radius + 1):
                    if k < 0 or k >= shape[2]:
                        continue
                    indices.append(i * shape[1] + j * shape[2] + k)
    return indices


def mse_computation(seq, learned_seq):
    common = common_nonzero([seq, learned_seq])
    return np.mean(np.square(seq[common] - learned_seq[common]))


def evaluate_result(seq, learned_seq, tumour_indices=None, mris_shape=None, square_radius=2, round_fact=6,
                    multiplier=1):
    init = time.time()
    if seq.shape != learned_seq.shape:
        error("The shape of the target and learned sequencing are not the same.")

    smse = None
    tumour = None
    common = common_nonzero([seq, learned_seq])

    mse = np.mean(np.square(seq[common] - learned_seq[common]))
    relmse = mse / np.mean(np.square(seq[common]))
    smape = np.sum(np.abs(seq[common] - learned_seq[common])) / \
            np.sum(np.abs(learned_seq[common]) + np.abs(seq[common]))

    mse = round(mse * multiplier, round_fact)
    print("The mean squared error is " + str(mse) + ".")

    relmse = round(relmse * multiplier, round_fact)
    print("The ratio of MSE and all-zero MSE is " + str(relmse) + ".")

    smape = round(smape, round_fact)
    print("The symmetric mean absolute percentage error is " + str(smape) + ".")

    if mris_shape is not None:
        stored_smse = np.zeros((square_radius * 2 + 1) ** len(mris_shape))
        for pixel in common:
            square_indices = square_index(pixel, mris_shape, square_radius)
            if len(square_indices) != len(stored_smse):
                continue
            for j, q_index in enumerate(square_indices):
                stored_smse[j] += (learned_seq[pixel] - seq[q_index]) ** 2
        smse = (min(stored_smse) / len(common)) * multiplier
        smse = round(smse, round_fact)
        print("The shifted MSE is " + str(smse) + ".")

    if tumour_indices is not None:
        tumour = np.mean(np.square(seq[tumour_indices] - learned_seq[tumour_indices]))
        tumour = round(tumour * multiplier, round_fact)
        print("The mean squared error of the tumor is " + str(tumour) + ".")

    end = round(time.time() - init, 3)
    print_timestamped("Time spent computing the error for the current mapping: " + str(end) + "s.")
    return mse, relmse, tumour, smape, smse
