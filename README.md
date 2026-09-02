# QADTFSR: Quality-Assisted Domain Transfer for Fast Face Super-Resolution

## Env Requirements
* Python 3.10, Pytorch 2.1.1
```bash
conda create -n  QADTFSR python=3.10
conda activate QADTFSR
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt
```


## Dataset Preparation

### Training

Prepare a text file listing the paths to all high-resolution training images (512×512, PNG format):

```bash
find /path/to/training/images -name "*.png" > train_files.txt
```
Change the val dir_path in configs/test.yaml.
```yaml
data:
  val:
    params:
      dir_path: /path/to/CelebA_test_pair/lq
      extra_dir_path: /path/to/CelebA_test_pair/hq

```
Train the model
```bash
python main.py --save_dir train_result/xx  --cfg_path configs/train.yaml
```


### Evaluation

Change the dir_path in configs/test.yaml.

```yaml
data:
  val:
    params:
      dir_path: /path/to/test/lq 
```


```bash
python inference.py --save_dir train_result/celeba
```



## Acknowledgements

This project builds upon:
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [Resshift](https://github.com/zsyoaoa/resshift)