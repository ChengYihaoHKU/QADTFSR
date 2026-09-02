# generate training file list
find /path/to/training/images -name "*.png" > train_files.txt


# evaluate the model
python inference_test.py --save_dir train_results/xx

# train the model
python main.py --save_dir train_results/xx