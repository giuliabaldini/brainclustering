source ~/Documents/UniBonn/Thesis/brainclustering/venv/bin/activate

./install.sh

python3 search.py --experiment_name runtests --data_folder ../data/braindata_small --main_clusters 4 --sub_clusters 100 --n_images 5 --sliced --test_set testA
python3 train.py --experiment_name runtests --data_folder ../data/braindata_small --main_clusters 4 --sub_clusters 100 --sliced
python3 test.py --experiment_name runtests --data_folder ../data/braindata_small --main_clusters 4 --sub_clusters 100 --sliced --test_set testA