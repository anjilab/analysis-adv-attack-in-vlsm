# Fine-Tuning Vision Language Model and Analysis of Robustness Against Adversarial Attack in Medical Image Segmentation 



## Abstract
Adversarial attacks have been fairly explored for computer vision and vision-language models. However, the avenue of adversarial attack for the vision language segmentation models (VLSMs) is still under-explored, especially for medical image analysis. Thus, we will investigate the robustness of VLSMs against adversarial attacks for 2D medical images with different modalities, such as radiology, photography, endoscopy, etc. First, we will fine-tune pre-trained VLSMs for medical image segmentation with adapters.Then, we will employ adversarial attacks---projected gradient descent (PGD) and fast gradient sign method (FGSM)---on that fine-tuned model to determine its robustness against adversaries. We will report models' performance decline to analyze adversaries' impact.

## Table of Contents
- [Attack Architecture](#methodology)
- [Setup](#setup)
- [Finetuning](#finetuning)
- [Results](#results)

## Attack Architecture
<div style="text-align: center;">
  <img src="media/architecture.png" alt="Attack-arch" style="width: 80%;"/>
</div>

## Setup
Please refer to the [VLSM-adapter](https://github.com/naamiinepal/vlsm-adapter) repo for environment setup, pretrained model setup and dataset_preparation.


## Finetuning

If you need to run our fine-tuning models, you can use the provided script:
```source .venv/bin/activate ```
```bash -x scripts/vlsm_adapter/kvasir_polyp.sh ```

This script will start the fine-tuning process, which is essential for customizing the model for specific tasks. 
In the file, all of the methods have been defined as bash scripts.
<!-- For running inference, please update the defaults configs (such as `ckpt_path`, `models`, etc.) in `scripts/inference.py` to get the evulation metric or generate the output masks (in the original resolution). -->

## Results
<div style="text-align: center;">
  <img src="media/results.png" alt="Attack-Results-FGSM-KSAVIR" style="width: 80%;"/>
</div>

## Data
The data for the training can be obtained from here: ArXiv Link: [arxiv.org/abs/2405.06196](https://www.arxiv.org/abs/2405.06196)

### Acknowledgement
We would like to thank [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) for providing a modifiable framework for running multiple experiments while tracking the hyperparameters.