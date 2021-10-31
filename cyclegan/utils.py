import cv2
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import make_grid
from tqdm import tqdm

from .models import Generator
from .dataset import AnimeDataset, CELEBA
from .config import IMG_SIZE
import os
from albumentations.pytorch import ToTensorV2


def get_mean_std(loader):
    sum, sum_squared, num_batches = 0, 0, 0

    for data in tqdm(loader):
        data = data.cuda()
        sum += torch.mean(data, dim=[0, 2, 3])
        sum_squared += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = sum / num_batches
    mean_of_squared = sum_squared / num_batches

    std = torch.sqrt(mean_of_squared - mean**2)

    return mean, std


def main():
    # dataset = AnimeDataset("./images/*.jpg", (70, 70))
    tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (70, 70),
            interpolation=InterpolationMode.BICUBIC,
        ),
        torchvision.transforms.ToTensor()
    ])
    dataset = CELEBA("./img_align_celeba/*.jpg", (70, 70))
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=12)

    print(get_mean_std(loader))


def plot_to_tensorboard(
    writer,
    real,
    fake,
    tensorboard_step,
    loss_critic=None,
    loss_gen=None,
):
    if loss_critic is not None and loss_gen is not None:
        writer.add_scalar("Loss Critic",
                          loss_critic,
                          global_step=tensorboard_step)
        writer.add_scalar("Loss Generator",
                          loss_gen,
                          global_step=tensorboard_step)

    with torch.no_grad():
        img_grid_real = make_grid(real[:4], normalize=True)
        img_grid_fake = make_grid(fake[:4], normalize=True)

        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def get_predicted_image(input_file,
                        output_name,
                        device='cpu',
                        output='anime',
                        model_dir='../trained_models',
                        does_return=False):
    img = cv2.imread(input_file)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255
    img = ToTensorV2()(image=img)['image'].to(dtype=torch.float32).unsqueeze(
        dim=0).to(device)

    with torch.no_grad():
        model = Generator(in_channels=3, out_channels=3).to(device)
        model.eval()

        if output == 'anime':
            model.load(name_file=os.path.join(model_dir, 'anime_gen.pth.tar'))
        else:
            model.load(name_file=os.path.join(model_dir, 'celeb_gen.pth.tar'))

        out = model(img).squeeze().permute(1, 2, 0)
        out = out.numpy() * 255

        if does_return:
            return out

        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_name, out)


if __name__ == "__main__":
    # main()
    pass
