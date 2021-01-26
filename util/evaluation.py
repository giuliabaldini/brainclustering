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
                "TumourMSE",
                "scaled_MSE",
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
        print("Measures for the predicted images.")
        mse, tumour = evaluate_result(mri_dict['target'],
                                      mri_dict['learned_target'],
                                      tumour_indices=truth_nonzero)
        print("Measures for the predicted images after smoothing.")
        smooth_mse, smooth_tumour = evaluate_result(mri_dict['target'],
                                                    mri_dict['learned_target_smoothed'],
                                                    tumour_indices=truth_nonzero)
        print_timestamped("Computing MSE on the scaled data")
        # Scale data in 0,1 and compute everything again
        s_real = normalize_with_opt(mri_dict['target'], 0)
        s_predicted = normalize_with_opt(mri_dict['learned_target'], 0)
        s_predicted_smoothed = normalize_with_opt(mri_dict['learned_target_smoothed'], 0)
        print("Measures for the predicted images.")
        scaled_mse, scaled_tumour = evaluate_result(s_real,
                                                    s_predicted,
                                                    tumour_indices=truth_nonzero)
        print("Measures for the predicted images after smoothing.")
        s_smooth_mse, s_smooth_tumour = evaluate_result(s_real,
                                                        s_predicted_smoothed,
                                                        tumour_indices=truth_nonzero)
        smoothing = 0 if smoothing == "median" else 1
        if self.excel:
            self.print_to_excel([query_name, -1,
                                 mse, tumour, scaled_mse, scaled_tumour,
                                 ])
            self.print_to_excel([query_name, smoothing,
                                 smooth_mse, smooth_tumour, s_smooth_mse, s_smooth_tumour,
                                 ])

    def close(self):
        if self.excel:
            self.ff.close()
            print_timestamped("Saved in " + str(self.excel_filename))


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

    mse = round(mse * multiplier, round_fact)
    print("MSE: " + str(mse), end="")
    if tumour_indices is not None:
        tumour = (np.square(np.subtract(seq[tumour_indices], learned_seq[tumour_indices]))).mean()
        tumour = round(tumour * multiplier, round_fact)
        print(", MSE of the tumor area: " + str(tumour), end="")
    print(".")
    return mse, tumour
