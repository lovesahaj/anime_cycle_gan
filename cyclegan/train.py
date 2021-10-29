from .config import (
    LRN_RATE,
    IMG_CHANNELS,
    LOAD_MODEL,
    SAVE_MODEL,
    DEVICE,
    NUM_EPOCHS,
    NUM_FEATURES,
    DEPTH_OF_CRITIC,
    MODEL_DIR,
    IMG_SIZE,
)
import torch
import torch.nn as nn
from .dataset import getLoaders
from .losses import CycleConsistencyLoss, WLossGP_Critic, WLoss_Generator, IdentityLoss
from .models import Generator, Critic
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

# train class for gans
from .utils import plot_to_tensorboard


class GANTrain(nn.Module):
    def __init__(self, device):
        super(GANTrain, self).__init__()
        self.device = device
        self.net_celeb_G = Generator(IMG_CHANNELS, IMG_CHANNELS).to(device)
        self.net_anime_G = Generator(IMG_CHANNELS, IMG_CHANNELS).to(device)
        self.net_celeb_C = Critic(IMG_CHANNELS, DEPTH_OF_CRITIC,
                                  NUM_FEATURES).to(device)
        self.net_anime_C = Critic(IMG_CHANNELS, DEPTH_OF_CRITIC,
                                  NUM_FEATURES).to(device)

        self.optim_celeb_G = optim.Adam(self.net_celeb_G.parameters(),
                                        lr=LRN_RATE)
        self.optim_anime_G = optim.Adam(self.net_anime_G.parameters(),
                                        lr=LRN_RATE)
        self.optim_celeb_C = optim.Adam(self.net_celeb_C.parameters(),
                                        lr=LRN_RATE)
        self.optim_anime_C = optim.Adam(self.net_anime_C.parameters(),
                                        lr=LRN_RATE)

        self.criterion_cycle_anime_celeb = CycleConsistencyLoss(
            self.net_anime_G, self.net_celeb_G)
        self.criterion_cycle_celeb_anime = CycleConsistencyLoss(
            self.net_celeb_G, self.net_anime_G)
        self.criterion_id_anime = IdentityLoss(self.net_anime_G)
        self.criterion_id_celeb = IdentityLoss(self.net_celeb_G)
        self.criterion_w_celeb_G = WLoss_Generator()
        self.criterion_w_anime_G = WLoss_Generator()
        self.criterion_w_celeb_C = WLossGP_Critic(self.net_celeb_C)
        self.criterion_w_anime_C = WLossGP_Critic(self.net_anime_C)

        self.anime_loader, self.celeb_loader = getLoaders(size=(IMG_SIZE,
                                                                IMG_SIZE))

        self.writer = SummaryWriter("logs/cycleGAN")
        self.tensorboard_step = 0

        if LOAD_MODEL:
            self.__load_checkpoint__()

        self.net_anime_C.train()
        self.net_celeb_C.train()
        self.net_celeb_G.train()
        self.net_celeb_G.train()

    def __save_checkpoint__(self):
        print("=>> Saving Checkpoint")
        self.net_celeb_G.save(self.optim_celeb_G,
                              name_file=os.path.join(MODEL_DIR,
                                                     'celeb_gen.pth.tar'))
        self.net_anime_G.save(self.optim_anime_G,
                              name_file=os.path.join(MODEL_DIR,
                                                     'anime_gen.pth.tar'))
        self.net_celeb_C.save(self.optim_celeb_C,
                              name_file=os.path.join(MODEL_DIR,
                                                     'celeb_cri.pth.tar'))
        self.net_anime_C.save(self.optim_anime_C,
                              name_file=os.path.join(MODEL_DIR,
                                                     'anime_cri.pth.tar'))

    def __load_checkpoint__(self):
        self.net_celeb_G.load(self.optim_celeb_G,
                              name_file=os.path.join(MODEL_DIR,
                                                     'celeb_gen.pth.tar'))
        self.net_anime_G.load(self.optim_anime_G,
                              name_file=os.path.join(MODEL_DIR,
                                                     'anime_gen.pth.tar'))
        self.net_celeb_C.load(self.optim_celeb_C,
                              name_file=os.path.join(MODEL_DIR,
                                                     'celeb_cri.pth.tar'))
        self.net_anime_C.load(self.optim_anime_C,
                              name_file=os.path.join(MODEL_DIR,
                                                     'anime_cri.pth.tar'))

    def __train_one__(self):
        loop = tqdm(zip(self.anime_loader, self.celeb_loader), leave=True)
        batch_idx = 0

        for real_anime, real_celeb in loop:
            self.optim_celeb_C.zero_grad()
            self.optim_anime_C.zero_grad()

            ## CRITIC LOSS ##
            real_celeb = real_celeb.to(self.device)
            real_anime = real_anime.to(self.device)

            # print(real_celeb.shape)
            # print(real_anime.shape)

            # FOR CELEBS
            fake_celeb = self.net_celeb_G(real_anime)
            critic_out_fake_celeb = self.net_celeb_C(fake_celeb.detach())
            critic_out_real_celeb = self.net_celeb_C(real_celeb)

            loss_celeb_critic = self.criterion_w_celeb_C(
                prediction_real=critic_out_real_celeb,
                prediction_fake=critic_out_fake_celeb,
                real=real_celeb,
                fake=fake_celeb,
            )
            ccloss_celeb = self.criterion_cycle_celeb_anime(real_celeb)
            idloss_celeb = self.criterion_id_celeb(real_celeb)

            sum_loss_celeb = loss_celeb_critic + ccloss_celeb + idloss_celeb

            sum_loss_celeb.backward(retain_graph=True)
            self.optim_celeb_C.step()

            # FOR ANIMES
            fake_anime = self.net_anime_G(real_celeb)
            critic_out_real_anime = self.net_anime_C(real_anime)
            critic_out_fake_anime = self.net_anime_C(fake_anime.detach())

            loss_anime_critic = self.criterion_w_anime_C(
                prediction_real=critic_out_real_anime,
                prediction_fake=critic_out_fake_anime,
                real=real_anime,
                fake=fake_anime,
            )

            ccloss_anime = self.criterion_cycle_anime_celeb(real_anime)
            idloss_anime = self.criterion_id_anime(real_anime)

            sum_loss_anime = loss_anime_critic + ccloss_anime + idloss_anime

            sum_loss_anime.backward(retain_graph=True)
            self.optim_anime_C.step()

            loss_critic = sum_loss_anime + sum_loss_anime

            ## GENERATOR LOSS ##

            # For Celebs
            self.optim_celeb_G.zero_grad()
            gen_celeb_fake = self.net_celeb_C(fake_celeb)
            loss_gen_celeb = self.criterion_w_celeb_G(gen_celeb_fake)

            loss_gen_celeb.backward()
            self.optim_celeb_G.step()

            # For Anime
            self.optim_anime_G.zero_grad()
            gen_anime_fake = self.net_anime_C(fake_anime)
            loss_gen_anime = self.criterion_w_anime_G(gen_anime_fake)

            loss_gen_anime.backward()
            self.optim_anime_G.step()

            loss_gen = loss_gen_anime + loss_gen_celeb

            if batch_idx % 5 == 0:
                with torch.no_grad():
                    fake_celeb = self.net_celeb_G(real_anime)
                    fake_anime = self.net_anime_G(real_celeb)
                    plot_to_tensorboard(
                        self.writer,
                        real_anime.detach(),
                        fake_celeb.detach(),
                        self.tensorboard_step,
                        loss_critic.item(),
                        loss_gen.item(),
                    )
                    plot_to_tensorboard(
                        self.writer,
                        real_celeb.detach(),
                        fake_anime.detach(),
                        self.tensorboard_step,
                        None,
                        None,
                    )

                    self.tensorboard_step += 1

            loop.set_postfix(loss_critic=loss_critic.item(),
                             loss_gen=loss_gen.item())
            batch_idx += 1

    def forward(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1} / {num_epochs}]")
            self.__train_one__()

            if SAVE_MODEL:
                self.__save_checkpoint__()


def main():
    trainObj = GANTrain(DEVICE)
    trainObj(NUM_EPOCHS)


if __name__ == '__main__':
    main()
