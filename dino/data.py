from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation


def load_seg_dataloader(img_dim=(256, 256)):
    # Transformations for images and masks
    transform = transforms.Compose(
        [transforms.Resize(img_dim), transforms.ToTensor()]
    )

    # Load the Cityscapes dataset
    dataset = VOCSegmentation(
        root="./data",
        year="2012",
        image_set="train",
        download=True,
        transform=transform,
        target_transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    return dataloader
