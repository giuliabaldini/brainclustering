import numpy as np

from util.util import error, print_timestamped, nonzero_union, normalize_with_opt


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
                "sMAPE",
                "TumourMSE",
                "scaled_MSE",
                "scaled_relMSE",
                "scaled_sMAPE",
                "scaled_TumourMSE"
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
        mse, relmse, smape, tumour = evaluate_result(mri_dict['target'],
                                                     mri_dict['learned_target'],
                                                     tumour_indices=truth_nonzero)
        smooth_mse, smooth_relmse, smooth_smape, smooth_tumour = evaluate_result(mri_dict['target'],
                                                                                 mri_dict['learned_target_smoothed'],
                                                                                 tumour_indices=truth_nonzero)
        print_timestamped("Computing MSE on the scaled data")
        # Scale data in 0,1 and compute everything again
        s_real = normalize_with_opt(mri_dict['target'], 0)
        s_predicted = normalize_with_opt(mri_dict['learned_target'], 0)
        s_predicted_smoothed = normalize_with_opt(mri_dict['learned_target_smoothed'], 0)
        scaled_mse, scaled_relmse, scaled_smape, scaled_tumour = evaluate_result(s_real,
                                                                                 s_predicted,
                                                                                 tumour_indices=truth_nonzero)
        s_smooth_mse, s_smooth_relmse, s_smooth_smape, s_smooth_tumour = evaluate_result(s_real,
                                                                                         s_predicted_smoothed,
                                                                                         tumour_indices=truth_nonzero)
        smoothing = 0 if smoothing == "median" else 1
        if self.excel:
            self.print_to_excel([query_name, -1,
                                 mse, relmse, smape, tumour,
                                 scaled_mse, scaled_relmse, scaled_smape, scaled_tumour,
                                 ])
            self.print_to_excel([query_name, smoothing,
                                 smooth_mse, smooth_relmse, smooth_tumour, smooth_smape,
                                 s_smooth_mse, s_smooth_relmse, s_smooth_tumour, s_smooth_smape,
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


def evaluate_result(seq, learned_seq, tumour_indices=None, round_fact=6, multiplier=1):
    if seq.shape != learned_seq.shape:
        error("The shape of the target and learned sequencing are not the same.")
    elif len(seq.shape) > 1:
        error("The evaluation is perfomed only on 1D arrays.")

    tumour = None
    nonzero_values = nonzero_union(seq, learned_seq)
    ground_truth = seq[nonzero_values]
    prediction = learned_seq[nonzero_values]

    # MSE: avg((A-B)^2)
    mse = (np.square(np.subtract(ground_truth, prediction))).mean()
    # RelMSE
    relmse = mse / np.square(ground_truth).mean()
    # SMAPE = 100/n sum |F - A| * 2/ sum |A| + |F|
    smape = (100 / np.size(ground_truth)) * \
            np.sum(np.abs(ground_truth - prediction) * 2 / (np.abs(prediction) + np.abs(ground_truth)))

    mse = round(mse * multiplier, round_fact)
    print("The mean squared error is " + str(mse) + ".")

    relmse = round(relmse * multiplier, round_fact)
    print("The ratio of MSE and all-zero MSE is " + str(relmse) + ".")

    smape = round(smape, round_fact)
    print("The symmetric mean absolute percentage error is " + str(smape) + ".")

    if tumour_indices is not None:
        tumour = (np.square(np.subtract(seq[tumour_indices], learned_seq[tumour_indices]))).mean()
        tumour = round(tumour * multiplier, round_fact)
        print("The mean squared error of the tumor is " + str(tumour) + ".")

    return mse, relmse, smape, tumour
