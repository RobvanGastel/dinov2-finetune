# Finetuning DINOv2 with LoRA for Image Segmentation

<p>
    <a href= "https://colab.research.google.com/github/RobvanGastel/dinov2-finetune/blob/main/Explanation.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>

This repository explores finetuning the DINOv2 (Oquab et al., 2024) encoder weights using Low-Rank Adaptation (Hu et al., 2021) (LoRA) and a simple 1x1 convolution decoder. LoRA makes it possible to finetune to new tasks easier without adjusting the original encoder weights by adding a small set of weights between each encoder block. The DINOv2 encoder weights are learned by self-supervised learning and capture the natural image domain accurately. For example, by just applying PCA to the outputs of the encoders we can already get a coarse segmentation of the objects in the image and see semanticly similar objects colored in the same color.

Check out the `Explanation.ipynb` notebook for a more detailed walkthrough of the code and ideas behind it.

![](/assets/examples/pca.png?raw=true)

## Setup
Install the packages using the `requirements.txt` file.

```bash
# using conda
conda create --name dino python=3.11
conda activate dino
# Install the package for dino_finetune imports,
pip install -e .
```

In the section below I explain all the flags used in the `main.py` to finetune to different datasets.

## Usage
An example to run finetuning on the VOC dataset with LoRA and a FPN decoder.

```bash
python main.py --exp_name base_voc --dataset voc --size base --use_lora --img_dim 308 308 --epochs 50 --use_fpn
```

**Flags**
Some explanation of the more useful flags to use when running experiments.
- --exp_name (str): The name of the experiment. This is used to identify the experiment and save results accordingly.
- --debug (flag): A boolean flag to indicate whether to debug the main.py training code.
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

**Pascal VOC**
I achieve a validation mean IoU of approximately 95% using LoRA and a 1x1 convolution decoder. When applying ImageNet-C corruptions (Hendrycks & Dietterich, 2019) to test robustness, the validation mean IoU drops to 88% with corruption severity level 5 (the maximum). The qualitative performance of this network is illustrated in the figure below. Based on their qualitative and quantitative performance, these pre-trained weights handle image corruptions effectively.

![](/assets/examples/voc_corruption_performance.png?raw=true)

You can use the pre-trained weights using the `--lora_weights` flag or just using the `load_parameters` function call. Registers here mean that extra context global context tokens are learned, see the second reference.


<table style="margin: auto">
  <thead>
    <tr>
      <th>finetuned components</th>
      <th>model</th>
      <th># of<br />params</th>
      <th>with<br />registers</th>
      <th>Pascal VOC<br />Validation mIoU</th>
      <th>Directory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1x1 Conv decoder</td>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">88.2%</td>
      <td>outputs/base_voc_no_lora.pt</td>
    </tr>
    <tr>
      <td>LoRA + 1x1 Conv decoder</td>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">95.3%</td>
      <td>outputs/base_voc.pt</td>
    </tr>
    <tr>
      <td>LoRA + FPN decoder</td>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">88.9%</td>
      <td>outputs/base_voc_fpn.pt</td>
    </tr>
  </tbody>
</table>

<br />

**ADE20k**
TODO


<table style="margin: auto">
  <thead>
    <tr>
      <th>finetuned components</th>
      <th>model</th>
      <th># of<br />params</th>
      <th>with<br />registers</th>
      <th>ADE20k<br />Validation mIoU</th>
      <th>Directory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1x1 Conv decoder</td>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LoRA + 1x1 Conv decoder</td>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>LoRA + FPN decoder</td>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>


## Citing
If you reference or use the codebase in your research, please cite:

```
@article{2024dinov2_lora_seg,
      title={Finetuning DINOv2 with LoRA for Image Segmentation},
      author={Rob van Gastel},
      year={2024}
    }
```

## References
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., … Bojanowski, P. (2024). DINOv2: Learning Robust Visual Features without Supervision (arXiv:2304.07193). arXiv. http://arxiv.org/abs/2304.07193

Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). Vision Transformers Need Registers (arXiv:2309.16588). arXiv. https://doi.org/10.48550/arXiv.2309.16588

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685). arXiv. http://arxiv.org/abs/2106.09685

Hendrycks, D., & Dietterich, T. G. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Surface Variations (arXiv:1807.01697). arXiv. https://doi.org/10.48550/arXiv.1807.01697