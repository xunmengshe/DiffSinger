# Usage of Refactor Branch
This is a cleaner version of Diffsinger, which provides:
- fewer code: scripts unused in the DiffSinger are marked **\*isolated\***;
- better readability: many important functions are annotated (however, **we assume the reader already knows how the neural networks work**);
- abstract classes: the bass classes are filtered out into the "basics/" folder and are annotated. Other classes inherent from the base classes.
- better file structre: tts-related files are filtered out into the "tts/" folder, as they are not used in DiffSinger.
- **(new) Much condensed version of the preprocessing, training, and inference pipeline**. The preprocessing pipeline is at 'preprocessing/opencpop.py', the training pipeline is at 'training/diffsinger.py', the inference pipeline is at 'inference/ds_cascade.py' or 'inference/ds_e2e.py'.

## Getting Started

### 0. Installation

```bash
# Install PyTorch manually (1.8.2 LTS recommended)
# See instructions at https://pytorch.org/get-started/locally/
# Below is an example for CUDA 11.1
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# Install other requirements
pip install -r requirements.txt
```

### 1. Preprocessing

```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/binarize.py --config configs/midi/cascade/opencs/ds1000.yaml
```
### 2. Training

```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/midi/cascade/opencs/ds1000.yaml --exp_name $MY_DS_EXP_NAME --reset  
```
### 3. Inference

```sh
CUDA_VISIBLE_DEVICES=0 python run.py --exp_name $MY_DS_EXP_NAME --infer
```
Easy inference with Google Colab:

Version 1: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kfmZ6Y018c5trSwQAbhdQtZ7Il8W_4BU)

Version 2: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V7yjNlh8_8o3IPK9buFb5MHVFrYmhELi)

# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 TTS](https://huggingface.co/spaces/NATSpeech/DiffSpeech)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)


This repository is the official PyTorch implementation of our AAAI-2022 [paper](https://arxiv.org/abs/2105.02446), in which we propose DiffSinger (for Singing-Voice-Synthesis) and DiffSpeech (for Text-to-Speech).

<table style="width:100%">
  <tr>
    <th>DiffSinger/DiffSpeech at training</th>
    <th>DiffSinger/DiffSpeech at inference</th>
  </tr>
  <tr>
    <td><img src="docs/resources/model_a.png" alt="Training" height="300"></td>
    <td><img src="docs/resources/model_b.png" alt="Inference" height="300"></td>
  </tr>
</table>

:tada: :tada: :tada: **Updates**:
 - Sep.11, 2022: :electric_plug: [DiffSinger-PN](docs/README-SVS-opencpop-pndm.md). Add plug-in [PNDM](https://arxiv.org/abs/2202.09778), ICLR 2022 in our laboratory, to accelerate DiffSinger freely.
 - Jul.27, 2022: Update documents for [SVS](docs/README-SVS.md). Add easy inference [A](docs/README-SVS-opencpop-cascade.md#4-inference-from-raw-inputs) & [B](docs/README-SVS-opencpop-e2e.md#4-inference-from-raw-inputs); Add Interactive SVS running on [HuggingFace🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger).
 - Mar.2, 2022: MIDI-B-version.
 - Mar.1, 2022: [NeuralSVB](https://github.com/MoonInTheRiver/NeuralSVB), for singing voice beautifying, has been released.
 - Feb.13, 2022: [NATSpeech](https://github.com/NATSpeech/NATSpeech), the improved code framework, which contains the implementations of DiffSpeech and our NeurIPS-2021 work [PortaSpeech](https://openreview.net/forum?id=xmJsuh8xlq) has been released. 
 - Jan.29, 2022: support MIDI-A-version SVS.
 - Jan.13, 2022: support SVS, release PopCS dataset.
 - Dec.19, 2021: support TTS. [HuggingFace🤗 TTS](https://huggingface.co/spaces/NATSpeech/DiffSpeech)

:rocket: **News**: 
 - Feb.24, 2022: Our new work, NeuralSVB was accepted by ACL-2022 [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2202.13277). [Demo Page](https://neuralsvb.github.io).
 - Dec.01, 2021: DiffSinger was accepted by AAAI-2022.
 - Sep.29, 2021: Our recent work `PortaSpeech: Portable and High-Quality Generative Text-to-Speech` was accepted by NeurIPS-2021 [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2109.15166) .
 - May.06, 2021: We submitted DiffSinger to Arxiv [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446).

## Environments
```sh
conda create -n your_env_name python=3.8
source activate your_env_name 
pip install -r requirements_2080.txt   (GPU 2080Ti, CUDA 10.2)
or pip install -r requirements_3090.txt   (GPU 3090, CUDA 11.4)
```

## Documents
- [Run DiffSpeech (TTS version)](docs/README-TTS.md).
- [Run DiffSinger (SVS version)](docs/README-SVS.md).

## Tensorboard
```sh
tensorboard --logdir_spec exp_name
```
<table style="width:100%">
  <tr>
    <td><img src="docs/resources/tfb.png" alt="Tensorboard" height="250"></td>
  </tr>
</table>

## Audio Demos
Old audio samples can be found in our [demo page](https://diffsinger.github.io/). Audio samples generated by this repository are listed here:

### TTS audio samples
Speech samples (test set of LJSpeech) can be found in [demos_1213](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_1213). 

### SVS audio samples
Singing samples (test set of PopCS) can be found in [demos_0112](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_0112).

## Citation
    @article{liu2021diffsinger,
      title={Diffsinger: Singing voice synthesis via shallow diffusion mechanism},
      author={Liu, Jinglin and Li, Chengxi and Ren, Yi and Chen, Feiyang and Liu, Peng and Zhao, Zhou},
      journal={arXiv preprint arXiv:2105.02446},
      volume={2},
      year={2021}}


## Acknowledgements
Our codes are based on the following repos:
* [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
* [HifiGAN](https://github.com/jik876/hifi-gan)
* [espnet](https://github.com/espnet/espnet)
* [DiffWave](https://github.com/lmnt-com/diffwave)

Also thanks [Keon Lee](https://github.com/keonlee9420/DiffSinger) for fast implementation of our work.
