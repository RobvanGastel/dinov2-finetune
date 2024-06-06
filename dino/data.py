import cv2
import numpy as np
import albumentations as A

from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class PascalVOCDataset(VOCSegmentation):
    def __init__(
        self,
        root="./data",
        year="2012",
        image_set="train",
        download=True,
        transform=None,
    ):
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
        )
        self.transform = transform

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros(
            (height, width, len(VOC_COLORMAP)),
            dtype=np.float32,
        )
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(float)

        # Ignore background
        # segmentation_mask = segmentation_mask[..., 1:]
        return segmentation_mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask = self._convert_to_segmentation_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = np.moveaxis(mask, -1, 0)
        image = np.moveaxis(image, -1, 0) / 255
        return image, mask


def load_voc_dataloader(img_dim=(490, 490), batch_size=6):
    # Transformation pipeline
    transform = A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])

    dataset = PascalVOCDataset(
        root="./data",
        year="2012",
        image_set="train",
        download=False,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
