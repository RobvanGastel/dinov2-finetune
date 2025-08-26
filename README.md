**[2025-08-25] Added the ability to finetune DINOv3 encoders!**

# Finetuning DINOv2, DINOv3 with LoRA for Image Segmentation

<p>
    <a href= "https://colab.research.google.com/github/RobvanGastel/dinov2-finetune/blob/main/Explanation.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>

This repository explores finetuning DINOv3 (Siméoni et al., 2025) or DINOv2 (Oquab et al., 2024) encoder weights using Low-Rank Adaptation (Hu et al., 2021) (LoRA) and a simple 1x1 convolution decoder. LoRA makes it possible to finetune to new tasks easier without adjusting the original encoder weights by adding a small set of weights between each encoder block. The DINOv2, DINOv3 encoder weights are learned by self-supervised learning and accurately capture the natural image domain. For example, by applying PCA to the outputs of the encoders, we can get a coarse segmentation of the objects in the image and see semantically similar objects colored in the same color.

Check out the `Explanation.ipynb` notebook for a more detailed walkthrough of the code and ideas behind it.

**DINOv3.** Les noise is visible when comparing the PCA outputs from DINOv3 versus the previous DINOv2.
![](/assets/examples/pca_dinov3.png?raw=true)

**DINOv2.**
![](/assets/examples/pca.png?raw=true)

Additionally, I tested a more recent paper, FeatUp, which uses PCA and upsamples the embeddings in the feature space, producing higher-resolution output. See the `Embedding_visualization.ipynb`.


https://github.com/user-attachments/assets/89dfae81-2c21-48bc-8877-87d8b732f9f0


## Setup
Install the packages using the `requirements.txt` file.

```bash
# using conda
conda create --name dino python=3.11
conda activate dino
# Install the package for dino_finetune imports,
pip install -e .
```

Special dependency if you want to investigate the encoder features in higher resolution using [FeatUp](https://github.com/mhamilton723/FeatUp). I recreated methods to process videos and images in the notebook `Embedding_visualization.ipynb`. To run it yourself in the notebook you need to install the FeatUp directory, and as it uses a custom kernel you need to make sure all the CUDA environment variables are configured properly.
```bash
# For CUDA_HOME/nvcc, make sure you install the cudatoolkit-dev tools
conda install -c conda-forge cudatoolkit-dev -y
# Now you should be able to run, 
nvcc -V
# So you can set the CUDA_HOME path
export CUDA_HOME=$CONDA_PREFIX
# For the LD_LIBRARY_PATH install cudnn
conda install -c conda-forge cudnn
# And set the variable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rob/miniconda3/envs/dino/lib
```

In the section below I explain all the flags used in the `main.py` to finetune to different datasets.

## Usage
An example to run finetuning on the VOC dataset with LoRA and an FPN decoder, either DINOv3 or DINOv2.

```bash
python main.py --exp_name base_voc --dataset voc --size base --dino_type dinov3 --img_dim 308 308 --epochs 50 --use_fpn
```

**Flags**
Some explanation of the more useful flags to use when running experiments.
- --exp_name (str): The name of the experiment. This is used to identify the experiment and save results accordingly.
- --debug (flag): A boolean flag to indicate whether to debug the main.py training code.
- --dataset (str): The name of the dataset to use. either `voc` or `ade20k`
- --size (str): The size configuration for the DINOv2 backbone parameter `small`, `base`, `large`, or `giant`
- --r (int): the LoRA rank (r) parameter to determine the amount of parameters. Usually, a small value like 3-9.
- --use_lora (flag): A boolean flag indicating whether to use Low-Rank Adaptation (LoRA). If this flag is present, LoRA is used. 
- --dino_type (str): Pass the DINO version to use either `dinov2`, or `dinov3`.
- --use_fpn (flag): A boolean flag to indicate whether to use the FPN decoder.
- --lora_weights (str): Path to the file location to load the LoRA weights and decoder head from.
- --img_dim (tuple of int): The dimensions of the input images (height width). This should be specified as two integers. Example: 308 308. 
- --epochs (int): The number of training epochs. This determines how many times the model will pass through the entire training dataset. Example: 50. 

There are some more unnamed parameters for training like the learning rate and batch size.

## Results

**Pascal VOC** \
**DINOv2.** I achieve a validation mean IoU of approximately 85.2% using LoRA and a 1x1 convolution decoder. When applying ImageNet-C corruptions (Hendrycks & Dietterich, 2019) to test robustness on Pascal VOC, the validation mean IoU drops to 72.2% with corruption severity level 5 (the maximum). The qualitative performance of this network is illustrated in the figure below. Based on their qualitative and quantitative performance, these pre-trained weights handle image corruption effectively.

![](/assets/examples/voc_corruption_performance.png?raw=true)

You can use the pre-trained weights using the `--lora_weights` flag or using the `load_parameters` function call. Registers here mean that extra context global context tokens are learned, see the second reference. All models are finetuned for 100 epochs.


<table style="margin: auto">
  <thead>
    <tr>
      <th>finetuned components</th>
      <th>pre-training</th>
      <th>model</th>
      <th># of<br />params</th>
      <th>with<br />registers</th>
      <th>Pascal VOC<br />Validation mIoU</th>
      <th>Pascal VOC-C<br />level 5<br />Validation mIoU</th>
      <th>Directory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1x1 Conv decoder</td>
      <td>DINOv2</td>
      <td>ViT-L/14</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">70.9%</td>
      <td align="right">66.6%</td>
      <td>output/dinov2/base_voc_no_lora.pt</td>
    </tr>
    <tr>
      <td>LoRA + 1x1 Conv decoder</td>
      <td>DINOv3</td>
      <td>ViT-L/16</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">77.6%</td>
      <td align="right">-%</td>
      <td>output/dinov3/large_base_voc_lora.pt</td>
    </tr>
    <tr>
      <td>LoRA + 1x1 Conv decoder</td>
      <td>DINOv2</td>
      <td>ViT-L/14</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">85.2%</td>
      <td align="right">72.2%</td>
      <td>output/dinov2/large_base_voc_lora.pt</td>
    </tr>
    <tr>
      <td>LoRA + FPN decoder</td>
      <td>DINOv2</td>
      <td>ViT-L/14</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">74.1%</td>
      <td align="right">65.6%</td>
      <td>output/dinov2/large_voc_fpn.pt</td>
    </tr>
  </tbody>
</table>

<br />

**ADE20k** \
**DINOv2.** I achieve a validation mean IoU of approximately 62.2% using LoRA and a 1x1 convolution decoder. With ADE20k-C with corruption severity level 5, the validation mean IoU drops to 55.8%. The qualitative performance of this network is illustrated in the figure below. 

![](/assets/examples/ade20k_corruption_performance.png?raw=true)


<table style="margin: auto">
  <thead>
    <tr>
      <th>finetuned components</th>
      <th>pre-training</th>
      <th>model</th>
      <th># of<br />params</th>
      <th>with<br />registers</th>
      <th>ADE20k<br />Validation mIoU</th>
      <th>ADE20k-C<br />level 5<br />Validation mIoU</th>
      <th>Directory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1x1 Conv decoder</td>
      <td>DINOv2</td>
      <td>ViT-L/14</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">57.2%</td>
      <td align="right">54.4%</td>
      <td>output/dinov2/large_ade20k_no_lora.pt</td>
    </tr>
    <tr>
      <td>LoRA + 1x1 Conv decoder</td>
      <td>DINOv3</td>
      <td>ViT-L/16</td>
      <td align="right">300M</td>
      <td align="center">✅</td>
      <td align="right">63.9%</td>
      <td align="right">57.7%</td>
      <td>output/dinov3/large_ade20k_lora.pt</td>
    </tr>
    <tr>
      <td>LoRA + 1x1 Conv decoder</td>
      <td>DINOv2</td>
      <td>ViT-L/14</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">62.2%</td>
      <td align="right">55.8%</td>
      <td>output/dinov2/large_ade20k_lora.pt</td>
    </tr>
    <tr>
      <td>LoRA + FPN decoder</td>
      <td>DINOv2</td>
      <td>ViT-L/14</td>
      <td align="right">300 M</td>
      <td align="center">✅</td>
      <td align="right">62.0%</td>
      <td align="right">54.7%</td>
      <td>output/dinov2/large_ade20k_fpn.pt</td>
    </tr>
  </tbody>
</table>


## Citing
If you reference or use the codebase in your research, please cite:

```
@misc{2024dinov2_lora_seg,
      title={Finetuning DINOv2, DINOv3 with LoRA for Image Segmentation},
      author={Rob van Gastel},
      year={2024}
    }
```

## References
Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., … Bojanowski, P. (2025). DINOv3 (No. arXiv:2508.10104). arXiv. https://doi.org/10.48550/arXiv.2508.10104

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., … Bojanowski, P. (2024). DINOv2: Learning Robust Visual Features without Supervision (arXiv:2304.07193). arXiv. http://arxiv.org/abs/2304.07193

Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). Vision Transformers Need Registers (arXiv:2309.16588). arXiv. https://doi.org/10.48550/arXiv.2309.16588

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685). arXiv. http://arxiv.org/abs/2106.09685

Hendrycks, D., & Dietterich, T. G. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Surface Variations (arXiv:1807.01697). arXiv. https://doi.org/10.48550/arXiv.1807.01697
