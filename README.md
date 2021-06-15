# BrainClustering
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This software aims at reproducing MRI images given another image in a process called MRI-to-MRI translation.
It is part of my master thesis at the University of Bonn, which was carried out in a joint project with the University 
of Cologne with [Jun.-Prof. Dr. Melanie Schmidt](https://orcid.org/0000-0003-4856-3905)
as supervisor and from an idea of [Dr. Liliana Lourenco Caldeira](https://orcid.org/0000-0002-9530-5899) from the University Hospital of Cologne.

BrainClustering first runs the clustering algorithms of choice on the MRI scans we are interested in (source and target).
Then, it computes tables that summarize the values that are present in the MRI images. 
After repeating this process with many images, we obtain extensive tables that can be queried to compute a new image.

The program can be run in two modes.
One can either train our model with `python train.py` and then `python test.py` or in search mode with `python search.py`,
which trains the tables on a selected section of the images and also runs the testing process.

## Installation
This project is written in python3.7/8 and C++17. 
It requires some Boost libraries to work:

* [Boost.Python](https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/index.html)
* [Boost.NumPy](https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/numpy/index.html)
* [Boost.Serialization](https://www.boost.org/doc/libs/1_72_0/libs/serialization/doc/index.html)

[CMakeLists.txt](./CMakeLists.txt) should be changed according to the Boost and the python versions.

You can run
```
./install.sh
```
to compile the C++ files.
This script should also install the Boost libraries.
If this does not work, please install the libraries separately and then run the command.

It also requires some python libraries, which can be installed with
```
pip install requirements.txt
```

## Running the program
The programs can be run with their respective python command.

### Data and Folder Organization
This software works only with 3D NIfTI files and the MRI scans should be skull-stripped, registered to the same template and have the same shape.
Please make sure that either the MRI scans have a zero background, 
or that the background is the minimum value of the image and you run the commands with `--preprocess` flag 0 or 1.

The `/path/to/data` folder should contain the data organized in folders according to training and testing images.
For the training data, the folder should be called `train`, while any name can be chosen for the test/validation data.
These folders should contain one folder per patient, where the name of the folder is the patient identifier.
Each patient folder should contain the files `t1, t2, t1ce, flair` and `truth`. 
The file names do not necessarily have to have these names, 
but they should contain these strings. 
Obviously, if you are only interested in `t1` and `t2`, the other files are not needed.
The tumor ground truth `truth` is used to compute the MSE in the tumor area, but if it is not available it will be ignored.

```
/path/to/data
|
└───train
│   └───pat1
│   |   │   t1.nii
│   |   │   t2.nii
│   |   │   ...
|   |   |   truth.nii
|   |
│   └───pat2
│   |   │   t1.nii
│   |   │   t2.nii
│   |   │   ...
|   |   |   truth.nii
|   |
│   └─── ...
└───val
│   └───pat1
│   |   │   t1.nii
│   |   │   t2.nii
│   |   │   ...
|   |   |   truth.nii
|   |
│   └─── ...
```

### Variables
Of all variables, the only required one is `/path/to/data`, which is the path to the data folder.

An important choice is the number of main clusters and the number of sub clusters.
We suggest to use 100 sub clusters and a number of main clusters between 3 and 5.
The clustering method can be chosen among the following: `kmeans`, `nesting` and `agglomerative`.
`agglomerative` is slow and performs worst so we do not recommend this choice.
`kmeans` is the fastest.

Each of the following commands can be run on 3D images or 2D slices. The default is to run it on 3D images. To change this, add
`--sliced` to any command or change the second argument from `3` to `2` when running the bash script. To choose a slice to run the 
algorithms on, use the parameter `--chosen_slice` in the python script. 
At the moment only transverse slices are supported and the default slice is the middle transverse slice for images of size 240x240x155.

An important parameter is the reference image, which can be selected with `--labeled_filename`.
This should be the path to a 3D NIfTI image containing a reliable tissue segmentation
of a brain into white matter, grey matter and cerebrospinal fluid.
This will be used only in case the number of main clusters is three. 
Otherwise the first training image will be used as reference, which is the case also if `--labeled_filename` is not specified.

For more information about the available options please check [base_options.py](options/base_options.py)

### Train
If we want to create a mapping from t1 to t2 we can start the training with
```
python3 train.py --experiment_name test_name --data_folder /path/to/data --main_clusters 4 --sub_clusters 100 --mapping_source t1 --mapping_target t2 --method kmeans
```

The _ indicates that a reference image has not been selected.

### Test

Once the model has been trained, we can retrieve the results for the `test_name` experiment with
```
python3 test.py --experiment_name test_name --data_folder /path/to/data --main_clusters 4 --sub_clusters 100 --mapping_source t1 --mapping_target t2 --method kmeans --postprocess 0 --excel --model_phase train --excel --test_set name/to/test
```
making sure that the same clustering algorithm and the same numbers of clusters are being used.

The flag `--excel` prints the results of the MSE computation to a csv file. When using the bash script, this is the default settings.
Otherwise, the results are only printed to screen.

`name/to/test` is the name of the folder where the MRI images for testing are located. For example, if the files to test are in `/path/to/data/name/to/test`, then 
`name/to/test` should be used. The default value is `test`.

The variable `--postprocess` can be used to postprocess the image:
* **0**: scales the image to the [0, 1] range,
* **1**: standardizes the image to have zero mean and unit variance,
* **2**: scales the image the [-1, 1] range.

It is also possible to select a filter to apply to the resulting image with `--smoothing`. 
Both the original and the filtered image will be saved as output.
The options are `median` and `average`, where `median` applies the median filter while `average` applies a simple averaging filter.
The default filter is `median`.

### Search

It is also possible to train on a small set of images, only the ones with the smallest MSE to the selected source image.
In this case the patient name of the query has to be selected with `--query_filename`, which is the name of the patient 
folder in `name/to/test`
```
python3 search.py --experiment_name test_name --data_folder /path/to/data --main_clusters 4 --sub_clusters 100 --mapping_source t1 --mapping_target t2 --method kmeans --postprocess 0 --query_filename query_name --n_images 5 --excel --test_set name/to/test
```
`--n_images` selects how many images with the closest MSE will be selected.

### Complementary

Moreover, we also implemented a method that computes the complementary of the source image, which is particularly effective for the translation from t1 to t2.
```
python3 complementary.py --experiment_name test_name --data_folder /path/to/data --mapping_source t1 --mapping_target t2 --postprocess 0 --excel --test_set test
```
