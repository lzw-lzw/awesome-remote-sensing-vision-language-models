# Awesome remote sensing vision language models
This is a repository for visual language models in remote sensing, including advanced methods and commonly used datasets in different applications, such as image-text retrieval, visual question answering, pretraining, etc.

*If you find any relevant papers that are not included here, please feel free to pull requests at any time.*

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
## Table of Contents
* [Surveys](#surveys)
* [Remote Sensing Vision Language Model](#remote-sensing-vision-language-model)
* [Applications](#applications)
  * [Pretraining](#pretraining)
  * [Image Captioning](#image-captioning)
  * [Text-based Image Generation](#text-based-image-generation)
  * [Image text Retrieval](#image-text-retrieval)
  * [Visual Question Answering](#visual-question-answering)
  * [Visual Grounding](#visual-grounding)
  * [Scene Classification](#scene-classification)
  * [Object Detection](#object-detection)
  * [Semantic Segmentation](#semantic-segmentation)
  * [Others](#others)
* [Dataset](#dataset)
  * [Image Captioning dataset](#image-captioning-dataset)
  * [Text-based Image Generation dataset](#text-based-image-generation-dataset)
  * [Text-based Image Retrieval dataset](#text-based-image-retrieval-dataset)
  * [Visual Question Answering dataset](#visual-question-answering-dataset)
  * [Visual Grounding dataset](#visual-grounding-dataset)
  * [Scene Classification dataset](#scene-classification-dataset)
  * [Object Detection dataset](#object-detection-dataset)
  * [Semantic Segmentation dataset](#semantic-segmentation-dataset)

# Surveys
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[Vision-Language Models in Remote Sensing: Current Progress and Future Trends](https://arxiv.org/abs/2305.05726)|arxiv 2023|-|
[The Potential of Visual ChatGPT For Remote Sensing](https://arxiv.org/abs/2304.13009)|arxiv 2023|-|

# Remote Sensing Vision Language Model
| Paper                                             |  Published in | Code/Project|  
|---------------------------------------------------|:-------------:|:------------:|
[RSGPT: A Remote Sensing Vision Language Model and Benchmark](https://arxiv.org/abs/2307.15266)|arxiv 2023|[code](https://github.com/Lavender105/RSGPT)
|RemoteGLM|2023|[code](https://github.com/lzw-lzw/RemoteGLM)|

# Applications
## Pretraining
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[S-CLIP: Semi-supervised Vision-Language Pre-training using Few Specialist Captions](https://arxiv.org/abs/2305.14095)|arxiv 2023|[code](https://github.com/alinlab/s-clip)
[RemoteCLIP: A Vision Language Foundation Model for Remote Sensing](https://arxiv.org/abs/2306.11029)|arxiv 2023|[code](https://github.com/ChenDelong1999/RemoteCLIP)
[RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model](https://arxiv.org/abs/2306.11300)|arxiv 2023|[Project](https://github.com/om-ai-lab/RS5M)
  
## Image Captioning
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[Deep Semantic Understanding of High Resolution Remote Sensing Image](https://ieeexplore.ieee.org/abstract/document/7546397)|CITS 2016|-
[Can a Machine Generate Humanlike Language Descriptions for a Remote Sensing Image?](https://ieeexplore.ieee.org/abstract/document/7891049)|TGRS 2017|-
[Exploring models and data for remote sensing image caption generation](https://ieeexplore.ieee.org/abstract/document/8240966)|TGRS 2017|[code](https://github.com/201528014227051/RSICD_optimal)|
[Natural language escription of remote sensing images based on deep learning](https://ieeexplore.ieee.org/abstract/document/8128075)|IGARSS 2017|-
[Description Generation for Remote Sensing Images Using Attribute Attention Mechanism](https://www.mdpi.com/2072-4292/11/6/612)|Remote Sensing 2019|-
[Vaa:Visual aligning attention model for remote sensing image captioning](https://ieeexplore.ieee.org/abstract/document/8843891)|IEEE Access 2019|-
[Exploring Multi-Level Attention and Semantic Relationship for Remote Sensing Image Captioning](https://ieeexplore.ieee.org/abstract/document/8943170)|IEEE Access 2019|-
[A multi-level attention model for remote sensing image captions](https://www.mdpi.com/2072-4292/12/6/939)|Remote Sensing 2020|-
[Remote sensing image captioning via variational autoencoder and reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S0950705120302586)|Knowledge-Based Systems 2020|-
[Truncation cross entropy loss for remote sensing image captionin](https://ieeexplore.ieee.org/abstract/document/9153154)|TGRS 2020|-
[Word–Sentence Framework for Remote Sensing Image Captioning](https://ieeexplore.ieee.org/abstract/document/9308980)|TGRS 2020|[code](https://github.com/hw2hwei/WordSent)|
[A novel SVM-based decoder for remote sensing image captioning](https://ieeexplore.ieee.org/abstract/document/9521989)|TGRS 2021|-
[High-resolution remote sensing image captioning based on structured attention](https://ieeexplore.ieee.org/abstract/document/9400386)|TGRS 2021|[code](https://github.com/Saketspradhan/High-Resolution-Remote-Sensing-Image-Captioning-Based-on-Structured-Attention)
[Exploring transformer and multilabel classification for remote sensing image captioning](https://ieeexplore.ieee.org/abstract/document/9855519)|GRSL 2022|-
[NWPU-captions dataset and mlca-net for remote sensing image captioning](https://ieeexplore.ieee.org/abstract/document/9866055)|TGRS 2022|-
[Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset](https://ieeexplore.ieee.org/abstract/document/9934924)|TGRS 2022|[code](https://github.com/Chen-Yang-Liu/RSICC)
[Transforming remote sensing images to textual descriptions](https://www.sciencedirect.com/science/article/pii/S0303243422000678)|INT J APPL EARTH OBS 2022|-
[Remote-sensing image captioning based on multilayer aggregated transformer](https://ieeexplore.ieee.org/abstract/document/9709791)|GRSL 2022|-
[Vlca: vision-language aligning model with cross-modal attention for bilingual remote sensing image captioning](https://ieeexplore.ieee.org/abstract/document/10066217)|J SYST ENG ELECTRON 2023|-
[Multi-source interactive stair attention for remote sensing image captioning](https://www.mdpi.com/2072-4292/15/3/579)|Remote Sensing 2023|-
[Changes to Captions: An Attentive Network for Remote Sensing Change Captioning](https://arxiv.org/abs/2304.01091)|arxiv 2023|[code](https://github.com/shizhenchang/chg2cap)

## Text-based Image Generation
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[Retro-Remote Sensing: Generating Images From Ancient Texts](https://ieeexplore.ieee.org/abstract/document/8660422)|J-STARS 2019|-
[Remote sensing image augmentation based on text description for waterside change detection](https://www.mdpi.com/2072-4292/13/10/1894)|Remote Sensing 2021|-
[Text-to-remote-sensing-image generation with structured generative adversarial networks](https://ieeexplore.ieee.org/abstract/document/9390223)|GRSL 2021|-
[Txt2img-MHN:Remote sensing image generation from text using modern hopfield network](https://arxiv.org/abs/2208.04441)|arxiv 2022|[code](https://github.com/YonghaoXu/Txt2Img-MHN)


## Image-text Retrieval
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[Textrs: Deep bidirectional triplet network for matching text to remote sensing images.](https://www.mdpi.com/2072-4292/12/3/405)|Remote Sensing 2020|-
[Deep unsupervised embedding for remote sensing image retrieval using textual cues](https://www.mdpi.com/2076-3417/10/24/8931)|Applied Sciences 2020|-
[A deep semantic alignment network for the cross-modal image-text retrieval in remote sensing](https://ieeexplore.ieee.org/abstract/document/9395191)|J-STARS 2021|-
[A lightweight multi-scale crossmodal text-image retrieval method in remote sensing](https://ieeexplore.ieee.org/abstract/document/9594840)|TGRS 2021|[code](https://github.com/xiaoyuan1996/retrievalSystem)
[Remote sensing cross-modal text-image retrieval based on global and local information](https://ieeexplore.ieee.org/abstract/document/9745546)|TGRS 2022|[code](https://github.com/xiaoyuan1996/GaLR)
[Multilanguage transformer for improved text to remote sensing image retrieval](https://ieeexplore.ieee.org/abstract/document/9925582)|J-STARS 2022|-
[Exploring a fine-grained multiscale method for cross-modal remote sensing image retrieva](https://arxiv.org/abs/2204.09868)|TGRS 2022|[code](https://github.com/xiaoyuan1996/AMFMN)
[Contrasting dual transformer architectures for multi-modal remote sensing image retrieval](https://www.mdpi.com/2076-3417/13/1/282)|Applied Sciences 2023|-
[Parameter-Efficient Transfer Learning for Remote Sensing Image-Text Retrieval](https://arxiv.org/abs/2308.12509)|arxiv 2023|-

## Visual Question Answering
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[RSVQA: Visual question answering for remote sensing data](https://ieeexplore.ieee.org/abstract/document/9088993)|TGRS 2020|[code](https://github.com/syvlo/RSVQA)
[Mutual Attention Inception Network for Remote Sensing Visual Question Answering](https://ieeexplore.ieee.org/document/9444570)|TGRS 2021|[code](https://github.com/spectralpublic/RSIVQA)
[How to find a good image-text embedding for remote sensing visual question answering?](https://arxiv.org/abs/2109.11848)|ECML-PKDD 2021|-
[Cross-Modal Visual Question Answering for Remote Sensing Data: The International Conference on Digital Image Computing: Techniques and Applications](https://ieeexplore.ieee.org/abstract/document/9647287)|DICTA 2021|-
[RSVQA meets bigearthnet: a new,large-scale, visual question answering dataset for remote sensing](https://ieeexplore.ieee.org/abstract/document/9553307)|IGARSS 2021|[code](https://github.com/syvlo/RSVQAxBEN)
[Self-Paced Curriculum Learning for Visual Question Answering on Remote Sensing Data](https://ieeexplore.ieee.org/abstract/document/9553624)|IGARSS 2021|-
[From easy to hard: Learning language-guided curriculum for visual question answering on remote sensing data](https://ieeexplore.ieee.org/abstract/document/9771224)|TGRS 2022|[code](https://github.com/YZHJessica/VQA-easy2hard)
[Language transformers for remote sensing visual question answering](https://ieeexplore.ieee.org/abstract/document/9884036)|IGARSS 2022|-
[Open-ended remote sensing visual question answering with transformers](https://www.tandfonline.com/doi/abs/10.1080/01431161.2022.2145583)|IJRS 2022|-
[Bi-modal transformer-based approach for visual question answering in remote sensing imagery](https://ieeexplore.ieee.org/abstract/document/9832935)|TGRS 2022|-
[Prompt-RSVQA: Prompting visual context to a language model for remote sensing visual question answering](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Chappuis_Prompt-RSVQA_Prompting_Visual_Context_to_a_Language_Model_for_Remote_CVPRW_2022_paper.html)|CVPRW 2022|-
[Change detection meets visual question answering](https://ieeexplore.ieee.org/abstract/document/9901476)|TGRS 2022|[code](https://github.com/YZHJessica/CDVQA)
[A spatial hierarchical reasoning network for remote sensing visual question answering](https://ieeexplore.ieee.org/abstract/document/10018408)|TGRS 2023|-
[Multilingual Augmentation for Robust Visual Question Answering in Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/10144189)|JURSE 2023|-
[LiT-4-RSVQA: Lightweight Transformer-based Visual Question Answering in Remote Sensing](https://arxiv.org/abs/2306.00758)|IGARSS 2023|[code](https://git.tu-berlin.de/rsim/lit4rsvqa)

## Visual Grounding
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Visual Grounding in Remote Sensing Images](https://dl.acm.org/doi/abs/10.1145/3503161.3548316)|ACMMM 2022|[data](https://sunyuxi.github.io/publication/GeoVG)|
|[RSVG: Exploring data and models for visual grounding on remote sensing data](https://ieeexplore.ieee.org/abstract/document/10056343)|TGRS 2023 |[code](https://github.com/ZhanYang-nwpu/RSVG-pytorch)

 
## Scene Classification
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[Zero-shot scene classification for high spatial resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/7902107)|TGRS 2017|-
[Fine-grained object recognition and zero-shot learning in remote sensing imagery](https://ieeexplore.ieee.org/abstract/document/8071030)|TGRS 2017|-
[Structural alignment based zero-shot classification for remote sensing scenes](https://ieeexplore.ieee.org/abstract/document/8645056)|ICECE 2018|-
[A distance-constrained semantic autoencoder for zero-shot remote sensing scene classification](https://ieeexplore.ieee.org/abstract/document/9633210)|J-STARS 2021|-
[Learning deep crossmodal embedding networks for zero-shot remote sensing image scene classification](https://ieeexplore.ieee.org/abstract/document/9321719)|TGRS 2021|-
[Generative adversarial networks for zero-shot remote sensing scene classification](https://www.mdpi.com/2076-3417/12/8/3760)|Applied Sciences 2022|-
[APPLeNet: Visual Attention Parameterized Prompt Learning for Few-Shot Remote Sensing Image Generalization using CLIP](https://arxiv.org/abs/2304.05995)|CVPR 2023|[code](https://github.com/mainaksingha01/APPLeNet)

## Object Detection
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[Text semantic fusion relation graph reasoning for few-shot object detection on remote sensing images](https://www.mdpi.com/2072-4292/15/5/1187)|Remote Sensing 2023|-
[Few-shot object detection in aerial imagery guided by textmodal knowledge](https://ieeexplore.ieee.org/abstract/document/10056362)|TGRS 2023|-

## Semantic Segmentation
| Paper                                             |  Published in | Code/Project|                                  
|---------------------------------------------------|:-------------:|:------------:|
[Semi-supervised contrastive learning for few-shot segmentation of remote sensing images](https://www.mdpi.com/2072-4292/14/17/4254)|Remote Sensing 2022|-
[Few-shot segmentation of remote sensing images using deep metric learning](https://ieeexplore.ieee.org/abstract/document/9721235)|GRSL 2022.
[Language-aware domain generalization network for cross-scene hyperspectral image classification](https://ieeexplore.ieee.org/abstract/document/10005113)|TGRS 2023|[code](https://github.com/YuxiangZhang-BIT/IEEE_TGRS_LDGnet)
[RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model](https://arxiv.org/abs/2306.16269)|arxiv 2023|[code](https://github.com/KyanChen/RSPrompter)
[RRSIS: Referring Remote Sensing Image Segmentation](https://arxiv.org/abs/2306.08625)|arxiv 2023|-


## Others



# Dataset
## Image Captioning Dataset
| Dataset                                            |  Home/Github | Download link|                                  
|---------------------------------------------------|:-------------:|:------------:|
[RSICD](https://ieeexplore.ieee.org/abstract/document/8240966)|[Github](https://github.com/201528014227051/RSICD_optimal)|[[BaiduYun]](https://pan.baidu.com/s/1bp71tE3#list/path=%2F)  [[Google Drive]](https://drive.google.com/open?id=0B1jt7lJDEXy3aE90cG9YSl9ScUk)
[Sydney-Captions](https://ieeexplore.ieee.org/abstract/document/7546397)|[Github](https://github.com/201528014227051/RSICD_optimal)|[[BaiduYun]](https://pan.baidu.com/s/1hujEmcG#list/path=%2F)
[UCM-Captions](https://ieeexplore.ieee.org/abstract/document/7546397)|[Github](https://github.com/201528014227051/RSICD_optimal)|[[BaiduYun]](https://pan.baidu.com/s/1mjPToHq#list/path=%2F)
[NWPU-RESISC45](https://ieeexplore.ieee.org/abstract/document/7891544)|[Github](https://gcheng-nwpu.github.io/#Datasets)|[[BaiduYun]](https://pan.baidu.com/s/1mifR6tU#list/path=%2F)  [[OneDrive]](https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs)
[DIOR-Captions](https://ieeexplore.ieee.org/abstract/document/10066217)|-|-
[RS-5M](https://github.com/om-ai-lab/RS5M)|[Github](https://github.com/om-ai-lab/RS5M)|[[HuggingFace]](https://huggingface.co/datasets/Zilun/RS5M/viewer/Zilun--RS5M/train?row=0)
[LEVIR-CC](https://ieeexplore.ieee.org/abstract/document/9934924)|[Github](https://github.com/Chen-Yang-Liu/RSICC)|[Google Drive](https://drive.google.com/drive/folders/1cEv-BXISfWjw1RTzL39uBojH7atjLdCG) |


## Text-based Image Generation Dataset

## Text-based Image Retrieval Dataset
| Dataset                                            |  Home/Project | Download link|                                  
|---------------------------------------------------|:-------------:|:------------:|
[RSITMD](https://arxiv.org/abs/2204.09868)|[Github](https://github.com/xiaoyuan1996/AMFMN)|[[BaiduYun]](https://pan.baidu.com/s/1gDj38mzUL-LmQX32PYxr0Q?pwd=NIST)  [[Google Drive]](https://drive.google.com/file/d/1NJY86TAAUd8BVs7hyteImv8I2_Lh95W6/view?usp=sharing)

## Visual Question Answering Dataset
| Dataset                                            |  Home/Project | Download link|                                  
|---------------------------------------------------|:-------------:|:------------:|
[RSVQA](https://ieeexplore.ieee.org/abstract/document/9088993)|[Home](https://github.com/syvlo/RSVQA)|[[data]](https://rsvqa.sylvainlobry.com/)
[RSVQA×BEN](https://ieeexplore.ieee.org/abstract/document/9553307)|[[Github]](https://github.com/syvlo/RSVQAxBEN) [[Home]](https://rsvqa.sylvainlobry.com/)|-
[RSIVQA](https://ieeexplore.ieee.org/document/9444570)|[Github](https://github.com/spectralpublic/RSIVQA)|-
[CDVQA](https://ieeexplore.ieee.org/abstract/document/9901476)|[Github](https://github.com/YZHJessica/CDVQA)|-

## Visual Grounding Dataset
| Dataset                                            |  Home/Project | Download link|                                  
|---------------------------------------------------|:-------------:|:------------:|
[DIOR-RSVG](https://ieeexplore.ieee.org/abstract/document/10056343) |[Github](https://github.com/ZhanYang-nwpu/RSVG-pytorch)|[[Google Drive]](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing)

## Scene Classification Dataset
| Dataset                                            |  Home/Project | Download link|                                  
|---------------------------------------------------|:-------------:|:------------:|
[NWPU-RESISC45](https://ieeexplore.ieee.org/abstract/document/7891544)|[Home](https://gcheng-nwpu.github.io/#Datasets)|[[OneDrive]](https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs)   [[BaiduYun]](https://pan.baidu.com/s/1mifR6tU)
[AID](https://ieeexplore.ieee.org/abstract/document/7907303)|[Home](https://captain-whu.github.io/AID/)|[[OneDrive]](https://1drv.ms/u/s!AthY3vMZmuxChNR0Co7QHpJ56M-SvQ)  [[BaiduYun]](https://pan.baidu.com/s/1mifOBv6#list/path=%2F)
[UC Merced Land-Use(UCM)](https://dl.acm.org/doi/abs/10.1145/1869790.1869829)|[Home](http://weegee.vision.ucmerced.edu/datasets/landuse.html)|-

## Object Detection Dataset
| Dataset                                            |  Home/Project | Download link|                                  
|---------------------------------------------------|:-------------:|:------------:|
[NWPU VHR-10](https://www.sciencedirect.com/science/article/abs/pii/S0924271614002524#preview-section-introduction)|[Home](https://gcheng-nwpu.github.io/#Datasets)|[[OneDrive]](https://1drv.ms/u/s!AmgKYzARBl5cczaUNysmiFRH4eE)   [[BaiduYun]](https://pan.baidu.com/s/1hqwzXeG#list/path=%2F) 
[DIOR](https://www.sciencedirect.com/science/article/abs/pii/S0924271619302825)|[Home](https://gcheng-nwpu.github.io/#Datasets)|[[Google Drive]](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC)    [[BaiduYun]](https://pan.baidu.com/s/1iLKT0JQoKXEJTGNxt5lSMg#list/path=%2F)
[FAIR1M](https://www.sciencedirect.com/science/article/abs/pii/S0924271621003269)|-|[[BaiduYun]](https://pan.baidu.com/share/init?surl=alWnbCbucLOQJJhi4WsZAw?pwd=u2xg)

## Semantic Segmentation Dataset
| Dataset                                            |  Home/Project | Download link|                                  
|---------------------------------------------------|:-------------:|:------------:|
Vaihingen|[Home](https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx)|[[BaiduYun]](https://pan.baidu.com/s/1EShNi22VfuIu3e6VygMb8g?pwd=3gsr)
Potsdam|[Home](https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx)|[[BaiduYun]](https://pan.baidu.com/s/13rdBXUN_ZdelWNlQZ3Y1TQ?pwd=6c3y)
Toronto|[Home](https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx)|-
[GID](https://www.sciencedirect.com/science/article/abs/pii/S0034425719303414)|[Home](https://x-ytong.github.io/project/GID.html)|[[BaiduYun code:GID5]](https://pan.baidu.com/s/1_DQluiDgJ4Z7dXSnciVx1A#list/path=%2F)   [[OneDrive]](https://whueducn-my.sharepoint.com/personal/xinyi_tong_whu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxinyi%5Ftong%5Fwhu%5Fedu%5Fcn%2FDocuments%2FGID&ga=1)
