# DINOv2 Finetuning
This repository explores finetuning DINOv2 (Oquab et al., 2024) encoder weights using Low-Rank Adaptation (Hu et al., 2021) (LoRA). The approach enhances the adaptation of the DINOv2 encoder for different downstream tasks with LoRA and a linear (1x1) convolutional head. Additionally, the DINOv2 encoder produces high-quality features making finetuning to new downstream tasks much easier. By applying PCA to the encoder features we can already get a coarse segmentation of the foreground object in the image.

Check out the `Explanation.ipynb` notebook for a more detailed walkthrough of the code.

![](/assets/examples/pca.png?raw=true)

## Setup
Install the packages using the `requirements.txt` file.

```bash
# using conda
conda create --name dino python=3.11
conda activate dino
conda install --file requirements.txt
# Install the package with,
pip install -e .
```

## Usage
An example to run finetuning on the VOC dataset with LoRA and a FPN decoder.

```bash
python main.py --exp_name base_voc --dataset voc --size base --use_lora --img_dim 308 308 --epochs 50 --use_fpn
```

**Flags**
Some explanation of the more useful flags to use when running experiments.
- --exp_name (str): The name of the experiment. This is used to identify the experiment and save results accordingly.
- --dataset (str): The name of the dataset to use. either `voc` or `ade20k`
- --size (str): The size configuration for the DINOv2 backbone parameter `small`, `base`, `large`, or `giant`
- --r (int): the LoRA rank (r) parameter to determine the amount of parameters. Usually, a small value like 3-9.
- --use_lora (flag): A boolean flag indicating whether to use Low-Rank Adaptation (LoRA). If this flag is present, LoRA is used. 
- --use_fpn (flag): A boolean flag to indicate whether to use the FPN decoder.
- --lora_weights (str): Path to the file location to load the LoRA weights en decoder head from.
- --img_dim (tuple of int): The dimensions of the input images (height width). This should be specified as two integers. Example: 308 308. 
- --epochs (int): The number of training epochs. This determines how many times the model will pass through the entire training dataset. Example: 50. 

There are some more unnamed parameters for training like the learning rate and batch size.

## Results
TODO


## References
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., … Bojanowski, P. (2024). DINOv2: Learning Robust Visual Features without Supervision (arXiv:2304.07193). arXiv. http://arxiv.org/abs/2304.07193

Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). Vision Transformers Need Registers (arXiv:2309.16588). arXiv. https://doi.org/10.48550/arXiv.2309.16588

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685). arXiv. http://arxiv.org/abs/2106.09685

