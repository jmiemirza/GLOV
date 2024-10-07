# GLOV: Guided Large Language Models as Implicit Optimizers for Vision Language Models

## Installation

#### 1.1 Installing dassl 
Our code is built upon the official codebase of [CoOp](https://github.dev/KaiyangZhou/CoOp).

As a first step, install `dassl` library (under `code/`) in your environment by following the instructions [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation).

To further install all other dependencies, please run the following command, after having your environment activated:

```
pip install -r requirements.txt
```
#### 1.2 Installing LLaVA-NeXT

In the same environemnt and directory you will also need to install LLaVA-NeXT with these [instruction](https://github.com/LLaVA-VL/LLaVA-NeXT?tab=readme-ov-file#installation).


## Datasets

Under `code/` first make an empty data folder: 

```
mkdir data
```

Then download and structure your datasets according to the instructions provided in 
the [CoOp](https://github.dev/KaiyangZhou/CoOp)
official repository. 
Most of the datasets are already implemented in their codebase. 
For other datasets, you will need to download the datasets from the official sources and structure them as the other 
datasets in the `CoOp` codebase. For convenience, we provide the download links for remaining datasets here: 

1. [resisc45](https://meta-album.github.io/datasets/RESISC.html)
2. [oxfordpets](https://www.robots.ox.ac.uk/~vgg/data/pets/)


## Experiments

To reproduce the results presented in Table 1 & 2, please run the following commands.  


#### 1. CLIP Results

1.1 With already discovered prompts (also present in the Appendix):

 
```  
bash scripts/zs_clip.sh 0 glov eurosat \
                           imagenet_r oxford_flowers \
                           imagenet_sketch dtd fgvc_aircraft food101 caltech101 \
                           imagenet stanford_cars sun397 imagenetv2 \
                           oxford_pets ucf101 resisc

```

Replace `0` with the desired GPU device number, `glov` with `glov_wo_guidance` for results without guidance, or `s_temp` for results with default template, `a photo of a {}`.

1.2 For starting a new optimization run with CLIP-ViT-B32:

```  
bash scripts/clip_opt.sh 0 eurosat \
                           imagenet_r oxford_flowers \
                           imagenet_sketch dtd fgvc_aircraft food101 caltech101 \
                           imagenet stanford_cars sun397 imagenetv2 \
                           oxford_pets ucf101 resisc

```

#### 2. LlaVa-OV Results

2.1 With already discovered prompts (also present in the Appendix):

```  
bash scripts/zs_llava.sh 0 glov eurosat \
                           imagenet_r oxford_flowers \
                           imagenet_sketch dtd fgvc_aircraft food101 caltech101 \
                           imagenet stanford_cars sun397 imagenetv2 \
                           oxford_pets ucf101 resisc
```
Replace `0` with the desired `GPU device number`, `glov` with `glov_wo_guidance` for results without guidance, or `s_temp` for results with default template, `Describe the category present in this image briefly and also identify the name of the category present.`.

2.2 For starting a new optimization run:

```  
bash scripts/llava_opt.sh 0 eurosat \
                           imagenet_r oxford_flowers \
                           imagenet_sketch dtd fgvc_aircraft food101 caltech101 \
                           imagenet stanford_cars sun397 imagenetv2 \
                           oxford_pets ucf101 resisc

```

### To cite us: 
```bibtex
@article{mirza2024glov,
    author    = {Mirza, M. Jehanzeb and Zhao, Mengjie and Mao, Zhuoyuan and Doveh, Sivan and Lin, Wei and Gavrikov, Paul and Dorkenwald, Michael and Yang, Shiqi and Jha, Saurav and Wakaki, Hiromi and Mitsufuji, Yuki and Possegger, Horst and Feris, Rogerio and Karlinsky, Leonid and Glass, James},
    journal   = {ArXiv},
    title     = {GLOV: Guided Large Language Models as Implicit Optimizers for Vision Language Models},
    year      = {2024}
    }
