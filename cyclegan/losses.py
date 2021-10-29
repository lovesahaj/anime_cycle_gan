from .config import DEVICE, LAMBDA_GP
import torch
import torch.nn as nn


class CycleConsistencyLoss(nn.Module):
    def __init__(self, gen1, gen2):
        super(CycleConsistencyLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.gen1 = gen1
        self.gen2 = gen2

    def forward(self, real):
        return self.l1loss(real, self.gen2(self.gen1(real)))


class WLossGP_Critic(nn.Module):
    def __init__(self, critic, λ=LAMBDA_GP):
        super(WLossGP_Critic, self).__init__()
        self.critic = critic
        self.λ = λ
        self.drift = 1e-3

    def __gp__(self, real, fake):
        batch, channels, height, width = real.shape

        ε = torch.randn((batch, 1, 1, 1)).repeat(
            (1, channels, height, width)).to(DEVICE)
        interpolated_image = real * ε + (fake * (1 - ε))
        mixed_score = self.critic(interpolated_image)

        gradient = torch.autograd.grad(
            inputs=interpolated_image,
            outputs=mixed_score,
            grad_outputs=torch.ones_like(mixed_score),
            retain_graph=True,
            create_graph=True)

        gradient = gradient[0].flatten(start_dim=1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1)**2)
        return penalty

    def forward(self, prediction_real, prediction_fake, real, fake):
        loss = torch.mean(prediction_fake) - torch.mean(prediction_real)
        gp = self.__gp__(real, fake)

        loss = loss + (gp * self.λ) + (self.drift *
                                       torch.mean(prediction_real**2))
        return loss


class WLoss_Generator(nn.Module):
    def __init__(self):
        super(WLoss_Generator, self).__init__()

    def forward(self, output):
        return -torch.mean(output)


class IdentityLoss(nn.Module):
    """docstring for IdentityLoss"""
    def __init__(self, gen):
        super(IdentityLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.gen = gen

    def forward(self, img):
        return self.l1loss(img, self.gen(img))
