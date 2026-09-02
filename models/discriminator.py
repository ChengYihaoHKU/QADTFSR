import abc
import torch
import numpy as np
from torch import nn
from typing import Tuple, List
import torch.nn.functional as F

class BaseDiscriminator(nn.Module):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Predict scores and get intermediate activations. Useful for feature matching loss
        :return tuple (scores, list of intermediate activations)
        """
        raise NotImplemented()
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseDiscriminator):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d,):
        super().__init__()
        self.channels = input_nc
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]
    
class Discriminator(NLayerDiscriminator):
    def __init__(self, input_nc=6, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d, weight=1.0, grad_weight=0.001, fm_weight=50.0):
        super().__init__(input_nc, ndf, n_layers, norm_layer)
        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layers = n_layers
        self.norm_layer = norm_layer
        self.bce_loss = torch.nn.BCEWithLogitsLoss(size_average=True,reduction='none')
        self.mse_loss = torch.nn.MSELoss(size_average=True,reduction='none')
        self.weight = weight
        self.grad_weight = grad_weight
        self.fm_weight = fm_weight
    def make_r1_gp(self, discr_real_pred, real_batch):
        if torch.is_grad_enabled():
            grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
        else:
            grad_penalty = 0
        real_batch = real_batch.detach()

        return grad_penalty


    def training_losses_sr(self, pred, gt, degrad):
        """
        Compute the training losses for the discriminator.
        
        :param x: Input tensor.
        :param y: Target tensor.
        :return: Loss value.
        """
        
        if self.input_nc == 6:
            gt = gt.requires_grad_(True)
            degrad = degrad.requires_grad_(True)
            x = torch.cat([gt, degrad], dim=1)
            
        # x.requires_grad = True
        
        scores_true, _ = self.forward(x)
        
        loss_grad = self.make_r1_gp(scores_true, x)
        
        y_true = torch.ones_like(scores_true)
        loss_true = self.bce_loss(scores_true, y_true)
        
        if self.input_nc == 6:
            x_r = torch.cat([pred.detach(), degrad], dim=1)
        scores_fake, _ = self.forward(x_r)
        y_fake = torch.zeros_like(scores_fake)
        loss_fake = self.bce_loss(scores_fake, y_fake)
        
        loss = (loss_true + loss_fake + loss_grad * self.grad_weight)
        
        return loss

    def gen_loss_sr(self, pred, gt, degrad):
        """
        Compute the training losses for the discriminator.
        
        :param x: Input tensor.
        :param y: Target tensor.
        :return: Loss value.
        """
        if self.input_nc == 6:
            x_r = torch.cat([pred, degrad], dim=1)
        scores_fake, fake_feature = self.forward(x_r)
        y_fake = torch.ones_like(scores_fake)
        loss = self.bce_loss(scores_fake, y_fake) * self.weight

        if self.input_nc == 6:
            x = torch.cat([gt, degrad], dim=1)
        _, gt_feature = self.forward(x)

        fm_loss = 0
        for real_f, fake_f in zip(gt_feature, fake_feature):
            fm_loss += self.mse_loss(real_f.detach(), fake_f)
        fm_loss = (fm_loss / len(gt_feature)) * self.fm_weight

        # print(f"Feature matching loss: {fm_loss.item()}, BCE loss: {loss.item()}")
        loss += fm_loss

        return loss