import albumentations as A


def get_corruption_transforms(img_dim: tuple[int, int], severity: int):
    """Augmentation pipeline to recreate the ImageNet-C dataset to evaluate the robustness of
    the DINOv2 backbone. Not all augmentations are available in Albumentations, so only the
    available augmentations are included if reasonable.

        Args:
            img_dim (tuple[int, int]): The height and width input tuple.
            severity (int): A severity level ranging from 1 to 5.

        Returns:
            A.Compose: An augmentation pipeline
    """
    return A.Compose(
        [
            A.OneOf(
                [
                    A.GaussNoise(
                        var_limit=[(0, 8), (8, 12), (12, 18), (18, 26), (26, 38)][
                            severity - 1
                        ],
                        always_apply=True,
                    ),
                    A.ISONoise(
                        intensity=[
                            (0.0, 0.3),
                            (0.5, 0.7),
                            (0.6, 0.9),
                            (0.6, 0.9),
                            (0.9, 1.2),
                        ][severity - 1],
                        always_apply=True,
                    ),
                    A.GaussianBlur(
                        blur_limit=[(0, 1), (1, 3), (1, 3), (3, 5), (5, 7)][
                            severity - 1
                        ],
                        always_apply=True,
                    ),
                    A.GlassBlur(
                        sigma=[0.7, 0.9, 1, 1.1, 1.5][severity - 1],
                        max_delta=[1, 2, 2, 3, 4][severity - 1],
                        iterations=[2, 1, 3, 2, 2][severity - 1],
                        always_apply=True,
                    ),
                    A.Defocus(
                        radius=[3, 4, 6, 8, 10][severity - 1],
                        alias_blur=[0.1, 0.5, 0.5, 0.5, 0.5][severity - 1],
                        always_apply=True,
                    ),
                    A.MotionBlur(
                        blur_limit=[3, 5, 5, 9, 13][severity - 1],
                        always_apply=True,
                    ),
                    A.ZoomBlur(
                        max_factor=[1.11, 1.16, 1.21, 1.26, 1.31][severity - 1],
                        always_apply=True,
                    ),
                    A.RandomSnow(
                        snow_point_upper=[0.1, 0.2, 0.45, 0.45, 0.55][severity - 1],
                        always_apply=True,
                    ),
                    A.ImageCompression(
                        quality_lower=[25, 18, 15, 10, 7][severity - 1],
                        quality_upper=[25, 18, 15, 10, 7][severity - 1],
                        always_apply=True,
                    ),
                    A.ElasticTransform(
                        alpha=[488, 288, 244 * 0.05, 244 * 0.07, 244 * 0.12][
                            severity - 1
                        ],
                        sigma=[
                            244 * 0.7,
                            244 * 0.08,
                            244 * 0.01,
                            244 * 0.01,
                            244 * 0.01,
                        ][severity - 1],
                        alpha_affine=[
                            244 * 0.1,
                            244 * 0.2,
                            244 * 0.02,
                            244 * 0.02,
                            244 * 0.02,
                        ][severity - 1],
                        always_apply=True,
                    ),
                ],
                p=1.0,
            ),
            A.Resize(height=img_dim[0], width=img_dim[1]),
        ]
    )
