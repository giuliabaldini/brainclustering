import kmeans1d
import leveled_kmeans
import mapping
import nesting
import numpy as np

from util.util import print_timestamped, info, remove_background, add_background, filter_blur, common_nonzero


class Mapping:
    def __init__(self, data_loader, plot_handler, model_folder, main_clusters, sub_clusters, method):
        self.data_loader = data_loader
        self.plot_handler = plot_handler
        self.model_folder = model_folder
        self.main_clusters = main_clusters
        self.sub_clusters = sub_clusters
        self.method = method
        self.table = None

    def cluster_mapping(self):
        self.table = mapping.Table(self.main_clusters, self.sub_clusters, 0.0)
        sub_clusters_d = int(self.sub_clusters / self.main_clusters)

        info("Mapping with clustering " + self.method + " with " + str(self.main_clusters) + " main clusters and " +
             str(self.sub_clusters) + " smaller clusters.")
        try:
            for i, filename in enumerate(self.data_loader.train_files):
                print_timestamped("Processing image " + filename.name + ".")
                current_mris = self.data_loader.return_file(filename)
                self.plot_handler.plot_train(current_mris,
                                             filename.name,
                                             self.data_loader.mri_shape,
                                             self.data_loader.affine)
                print_timestamped("Transformed data.")
                common_nonzero_train = common_nonzero([current_mris['source'], current_mris['target']])
                m1, _ = remove_background(current_mris['source'], nonzero_indices=common_nonzero_train)
                m2, _ = remove_background(current_mris['target'], nonzero_indices=common_nonzero_train)
                print_timestamped("Removed background.")
                k_list_aggl = [self.sub_clusters, self.main_clusters]
                if self.method == "nesting":
                    d1 = nesting.cluster1d(m1, k_list_aggl, 0)
                    d2 = nesting.cluster1d(m2, k_list_aggl, 0)
                else:
                    k_list_div = [self.main_clusters, sub_clusters_d]
                    d_d1 = leveled_kmeans.cluster1d(m1, k_list_div)
                    d_d2 = leveled_kmeans.cluster1d(m2, k_list_div)
                    print_timestamped("Computed kmeans dendograms for the two MRIs.")
                    d1 = leveled_kmeans.build_agglomerative_dendogram(d_d1, k_list_div)
                    d2 = leveled_kmeans.build_agglomerative_dendogram(d_d2, k_list_div)

                print_timestamped("Computed dendograms for the two MRIs.")
                # We find the mapping, but we only need to find it for d1, because then d2 will be mapped to d1
                self.table.add_entry(m1, d1, m2, d2)
                print_timestamped("Entry added to the tables.")
                curr_filename = self.model_folder / (
                        "model_" + self.method + "_main" + str(self.main_clusters) + "_sub" + str(
                    self.sub_clusters) + "_" + str(i))
                mapping.save_mapping(self.table, str(curr_filename))
                info("Model saved in " + str(curr_filename) + ".")
                if self.plot_handler.labels_folder is not None:
                    labels = np.zeros(m1.shape[0])
                    nesting.build_labels(labels, d1, k_list_aggl, 1, 1.0)
                    new_labels_m1 = add_background(labels, current_mris['source'].shape[0], common_nonzero_train)

                    labels = np.zeros(m2.shape[0])
                    nesting.build_labels(labels, d2, k_list_aggl, 1, 1.0)
                    new_labels_m2 = add_background(labels, current_mris['target'].shape[0], common_nonzero_train)
                    self.plot_handler.plot_shaded_labels(filename.name, new_labels_m1, new_labels_m2,
                                                         self.method, self.main_clusters, self.data_loader.mri_shape,
                                                         self.data_loader.affine)
        except MemoryError:
            info("The clustering with method " + self.method + " caused a memory error. ")

    def restore_table(self, model):
        self.table = mapping.Table(10, 2, 0.0)
        mapping.restore_mapping(self.table, str(model))

    def return_results_query(self, mri_dict, smoothing):
        # Find all the common indices between the query image and the labeled image
        query, query_nonzero = remove_background(mri_dict['source'])

        labels_m1, _ = kmeans1d.cluster(query, self.main_clusters)
        labels_m1 = np.array(labels_m1, dtype=np.float_)

        # print(np.unique(labels_m1, return_counts=True))
        # print(np.unique(mri_lab_query, return_counts=True))
        print_timestamped("Computed labels for query.")

        new_m2 = np.zeros(query.shape[0])
        self.table.return_sequencing(query, labels_m1, new_m2)
        mri_dict['learned_target'] = add_background(new_m2, mri_dict['source'].shape[0], query_nonzero)
        mri_dict['learned_target_smoothed'] = filter_blur(mri_dict['learned_target'], self.data_loader.mri_shape,
                                                          smoothing)
        print_timestamped("Output smoothed.")

        return mri_dict


def complementary_mapping(mri_dict, mri_shape, smoothing):
    query, query_nonzero = remove_background(mri_dict['source'])
    new_target = np.zeros(query.shape[0])
    max_pixel = query.max()
    min_pixel = query.min()
    min_max = min_pixel + max_pixel
    for p in range(query.shape[0]):
        new_target[p] = min_max - query[p]

    mri_dict['learned_target'] = add_background(new_target, mri_dict['source'].shape[0], query_nonzero)
    mri_dict['learned_target_smoothed'] = filter_blur(mri_dict['learned_target'], mri_shape, smoothing)
    return mri_dict
