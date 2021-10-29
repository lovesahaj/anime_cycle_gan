from .config import BATCH_SIZE
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


class AnimeDataset(Dataset):
    """
    Batch size 128 -> 840 MB
    Mean -> [0.6862, 0.5877, 0.5717]
    Stddev -> [0.2912, 0.2927, 0.2726]
    """
    def __init__(self, data_dir, size, transform=None) -> None:
        super(AnimeDataset, self).__init__()
        self.files = glob(data_dir)
        self.size = size
        self.tranform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = cv2.imread(self.files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC)
        image = image / 255

        if self.tranform is not None:
            image = self.tranform(image=image)["image"]
        else:
            image = ToTensorV2()(image=image)["image"].to(dtype=torch.float32)

        return image


class CELEBA(Dataset):
    """
    Batch size 512 -> 1049 MB
    Mean -> [0.5061, 0.4255, 0.3829]
    STDDEV -> [0.3105, 0.2902, 0.2895]
    """
    def __init__(self, data_dir, size, transform=None) -> None:
        super(CELEBA, self).__init__()
        self.files = glob(data_dir)
        self.size = size
        self.tranform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = cv2.imread(self.files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.size is not None:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC)
        image = image / 255

        if self.tranform is not None:
            # print(self.tranform(image=image))
            image = self.tranform(image=image)["image"]
        else:
            image = ToTensorV2()(image=image)["image"].to(dtype=torch.float32)

        return image


def getLoaders(size):
    transform_anime = A.Compose([
        A.Normalize(mean=[0.6862, 0.5877, 0.5717],
                    std=[0.3105, 0.2902, 0.2895]),
        ToTensorV2()
    ])
    transform_celeba = A.Compose([
        A.Normalize(mean=[0.5061, 0.4255, 0.3829],
                    std=[0.2912, 0.2927, 0.2726]),
        ToTensorV2()
    ])

    anime = AnimeDataset("./anime/*.jpg", size)
    celeb = CELEBA("./img_align_celeba/*.jpg", size)

    # print(len(anime))
    # print(len(celeb))
    celeb = random_split(celeb, [len(anime), len(celeb) - len(anime)])[0]

    animeLoader = DataLoader(
        anime,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
    )
    celebLoader = DataLoader(
        celeb,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
    )

    return animeLoader, celebLoader


def main():
    anime = AnimeDataset("./anime/*.jpg", (70, 70))
    # plt.imshow(anime[0].permute(1, 2, 0))
    # plt.show()
    # min = anime[0].shape[1]
    # max = anime[0].shape[1]
    # for i in tqdm(anime):
    #     if min > i.shape[1]:
    #         min = i.shape[1]
    #         print("Min:", i.shape)  # 25
    #     if max < i.shape[1]:
    #         max = i.shape[1]
    #         print("Max: ", i.shape)  # 220

    celeb = CELEBA("./img_align_celeba/*.jpg", size=(70, 70))
    celeb = random_split(celeb, [len(anime), len(celeb) - len(anime)])[0]

    # plt.imshow(celeb[0].permute(1, 2, 0))
    # plt.show()

    print(len(anime))
    print(len(celeb))


if __name__ == "__main__":
    main()
