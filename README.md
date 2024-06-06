# DINOv2 Finetuning
This repository explores fine-tuning DINOv2 encoder weights using LoRA (Low-Rank Adaptation). The approach enhances the adaptation of the DINOv2 encoder for different downstream tasks with LoRA and a linear (1x1) convolutional head. Additionally, the DINOv2 encoder produces high-quality features making finetuning to new downstream tasks much easier. By applying PCA to the encoder features we can already get a coarse segmentation of the foreground object in the image.

![](pca.png?raw=true)

## Setup
Install the packages using the `requirements.txt` file.

```bash
# using conda
conda create --name dino python=3.11
conda activate dino
conda install --file requirements.txt
# Or pip
pip install requirements.txt
```

## Usage

## Results
finetuning with Linear classifier versus linear classifier with LoRA  

## References