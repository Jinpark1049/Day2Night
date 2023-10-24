# Day2Night

Day2Night image2image translation using CycleGAN + StyleLoss
Image-to-Image translation using Gan based models, currently n2d_v3 is the most up-to-date module to run the models.
 

## CycleGAN
PyTorch implementation of [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)
Instead of using Cycle-Consistency Loss, replaced it by [Style Loss](https://ieeexplore.ieee.org/document/7780634), also adapted upsampling instead of deconvolutional layer.
+ 9 residual blocks for generator

## Adain
Pytorch implementation of [Adain](https://arxiv.org/abs/1703.06868)

## Datasets
Build your own dataset by setting up the following directory structure:

    ├── datasets                   
    |   ├── <dataset_name>         # i.e. Day2Night
    |   |   ├── train              # Training
    |   |   |   ├── A              # Contains domain A images (i.e. Day)
    |   |   |   └── B              # Contains domain B images (i.e. Night)
    |   |   └── test               # Testing
    |   |   |   ├── A              # Contains domain A images (i.e. Day)
    |   |   |   └── B              # Contains domain B images (i.e. Night)
    
* BDD100K can be downloaded from here: [BDD100K](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)
* Berkeley Deepdrive
* Nexet 2017
* VFP290K

## Getting Started

### Model Selection

```bash
n2d_v3/options/base_options.py
```
Specify the '--model_name' for "Adain" or "cycleGAN"


### Train with DDP

- Run the following command:
```bash
torchrun --nnodes=1 --nproc-per-node=4 main.py --distributed
```

### Train with DDP & Wandb

- Run the following command:
```bash
wandb login
```
```bash
torchrun --nnodes=1 --nproc-per-node=4 main_wb.py --distributed --wandb
```
Click here to view the experimental results [D2N experimental results](https://wandb.ai/parkjy2/d2n_gan/runs/e870uysj?workspace=user-parkjy2)

### Test 

- For test, specify which model to use from --model_name ex) cycleGAN, Adain
- Also, specify the experimental name from --name to load pt files.
- pt file need to be in experiment name folder. ex) /checkpoints/cycleGAN_StyleLoss_256x256/weight_200.pt 
```bash
python test.py
```


