# CORE-ReID V2 
**CORE-ReID V2 : Advancing the Domain Adaptation for Object Re-Identification with Optimized Training and Ensemble Fusion**


<img src="resource\people_dance.gif" height="270" /> <img src="resource\car_dance.gif" height="270" />

- ❤️Our project page: https://trinhquocnguyen.github.io/core-reid-v2-homepage/
- ❤️Paper: https://www.mdpi.com/3042-5999/1/1/4
- ❤️About Trinh: https://www.linkedin.com/in/trinh-quoc-nguyen/


**[2025/07/04: Good News!]** ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)    

* Our paper got accepted by MDPI Journal.

## TODO
- [X] Explain step by step how to run the source code with tutorial videos
- [X] Update Readme.md file
- [X] Fix the absolute path
- [X] Initialize the source code

## Updates
- **[2024/07/03: Source code is released!]** ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)
    * The first version of source code has been initialized.

## (I) Installation
This codebase has been developed with python version 3.8, PyTorch version 1.8.1, CUDA 12.1 and torchvision 0.9.1
- Installation
    - Run: ```conda env create -f core_reid_v2.yml```
- We used serveral servers to train the models:
    - NVIDIA Quadro RTX 8000 x 2


## (II) Training 
### 1. Train CycleGAN models
- For Person ReID, Follow the guidance from: [camstyle-for-person-reid](https://github.com/TrinhQuocNguyen/camstyle-for-person-reid)
- For Vehicle ReID, Follow the guidance from: [pytorch-CycleGAN-and-pix2pix-vehicle-reid](https://github.com/TrinhQuocNguyen/pytorch-CycleGAN-and-pix2pix-vehicle-reid)

### 2. Generate the training datataset
- For Person ReID, Follow the guidance from: [camstyle-for-person-reid](https://github.com/TrinhQuocNguyen/camstyle-for-person-reid)
- For Vehicle ReID, Follow the guidance from: [pytorch-CycleGAN-and-pix2pix-vehicle-reid](https://github.com/TrinhQuocNguyen/pytorch-CycleGAN-and-pix2pix-vehicle-reid)

### 3. Train the ReID model 
- Modify the data path in the "global_config.yaml" file: 
```data_path: "/your/data/path/"```
- Reconfirm the train files in "scripts" folder
- For example, train CUKH03 => MARKET1501: 
```sh scripts/run_cuhk03np2market.sh```
#### 3.1 Step 1: Pretraining on Source Domain
For example, train CUKH03 => MARKET1501 with Resnet50 architecture:
```
CUDA_VISIBLE_DEVICES=2,3 python source_pretrain.py -ds cuhk03np -dt market \
                        --logs-dir logs/cuhk03np2market/resnet50/source_pretraining -b 128 --arch resnet50_source
```
#### 3.2 Step 2: Fine-Tuning on Target Domain
For example, fine-tune CUKH03 => MARKET1501 with Resnet50 architecture:
```
CUDA_VISIBLE_DEVICES=2,3 python target_train.py -dt market \
                       --logs-dir logs/cuhk03np2market/resnet50/900_ECAB1_LR \
                       --initial-weights logs/cuhk03np2market/resnet50/source_pretraining -b 128 \
                       --num-clusters 900 --arch resnet50
```
#### 3.3 Step 3: Evaluation on Target Domain
For example, test CUKH03 => MARKET1501 with Resnet50 architecture:
```
CUDA_VISIBLE_DEVICES=2,3 python model_test.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid --resume logs/cuhk03np2market/resnet50/900_ECAB1_LR/model_best.pth.tar --num-classes 900 --arch resnet50 
```

#### 3.4 Step 4: Visualize the heatmap by using Grad-CAM
For example, saving the heatmap of CUKH03 => MARKET1501 with Resnet50 architecture:
```
python model_save_feature_maps.py -dt market --data-dir /old/home/ccvn/Workspace/trinh/data/reid \
                                --resume logs/cuhk03np2market/resnet50/900_ECAB1_LR/model_best.pth.tar \
                                --num-classes 900 --arch resnet50 --batch-size 1

```

### 4. Tutorials
Watch the Tutorial: 

[![Watch the Tutorial](resource/youtube.png)](https://youtu.be/bVyPntMedLQ)

## Citations
Please cite our paper if you find it useful
```
@article{,
  author    = {Nguyen, Trinh Quoc and Prima, Oky Dicky Ardiansyah and Irfan, Syahid Al and Purnomo, Hindriyanto Dwi and Tanone, Radius},
  title     = {CORE-ReID V2: Advancing the Domain Adaptation for Object Re-Identification with Optimized Training and Ensemble Fusion},
  journal   = {AI Sensors},
  doi       = {10.3390/aisens1010004},
  volume    = {1},
  number    = {1},
  article-number = {4},
  year      = {2025},
  issn      = {3042-5999},
  url       = {https://www.mdpi.com/3042-5999/1/1/4},
}
```
## Acknowledgement
Thank you for great works below:
- [open-reid](https://github.com/Cysu/open-reid)
- [MEB-Net](https://github.com/YunpengZhai/MEB-Net)
- [LF2](https://github.com/DJEddyking/LF2)
- [CBAM](https://github.com/luuuyi/CBAM.PyTorch)
