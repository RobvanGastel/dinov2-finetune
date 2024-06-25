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
# Or pip
pip install requirements.txt
```

## Usage
TODO

## Results
TODO




## References
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., … Bojanowski, P. (2024). DINOv2: Learning Robust Visual Features without Supervision (arXiv:2304.07193). arXiv. http://arxiv.org/abs/2304.07193

Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). Vision Transformers Need Registers (arXiv:2309.16588). arXiv. https://doi.org/10.48550/arXiv.2309.16588

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685). arXiv. http://arxiv.org/abs/2106.09685

