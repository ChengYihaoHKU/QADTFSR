# QADTFSR: Quality-Assisted Domain Transfer for Fast Face Super-Resolution
🌐 [Project Page](https://chengyihaohku.github.io/QADTFSR/) | 📄 [Paper](https://ieeexplore.ieee.org/abstract/document/11420887/)
## PyTorch implementation of QADTFSR:
>Yi-Hao Cheng, Wan-Chi Siu, and Shing-Chow Chan, "Quality-Assisted Domain Transfer for Fast Face Super-Resolution," IEEE Signal Processing Letters, vol. 33, pp. 1386–1390, 2026, doi: 10.1109/LSP.2026.3669939.
## Env Requirements
* Python 3.10, Pytorch 2.1.1
```bash
conda create -n  QADTFSR python=3.10
conda activate QADTFSR
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt
```

## Pretrained Model

Download the pretrained model weights from the [Releases page](https://github.com/ChengYihaoHKU/QADTFSR/releases/tag/v1.0) and place it under the `weights/` directory.

**Option 1: Direct download (browser)**

Click the link below and save it as `weights/best.pth`:

[best.pth (629MB)](https://github.com/ChengYihaoHKU/QADTFSR/releases/download/v1.0/best.pth)

**Option 2: Command line**

```bash
mkdir -p weights
wget -O weights/best.pth https://github.com/ChengYihaoHKU/QADTFSR/releases/download/v1.0/best.pth
```

Or using the GitHub CLI:

```bash
gh release download v1.0 --pattern "best.pth" --dir weights
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

## Copyright

Copyright preserve by the University of Hong Kong and Saint Francis University.
Users can use this package for further development. However, it is not for commerical use.