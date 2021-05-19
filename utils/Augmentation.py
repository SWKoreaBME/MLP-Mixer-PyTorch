import albumentations as album


class Transform:
    def __init__(self, phase='train'):
        self.phase = phase
        if self.phase == "train":
            self.data_augmentation = album.Compose([
                album.OneOf([
                    album.HorizontalFlip(p=0.5),
                    album.VerticalFlip(p=0.5),
                    album.GaussNoise(var_limit=(0, 1), p=0.5),
                    album.Rotate(limit=180, p=0.5),
                    album.ElasticTransform(p=0.5, approximate=True)
                ])
            ], p=0.8)

        elif self.phase == "test":
            self.data_augmentation = album.Compose([
            ])


if __name__ == '__main__':
    pass
