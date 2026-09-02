import math
import abc
import abc
import torch
import numpy as np
import numpy as np
from torch import nn
from einops import rearrange
from inspect import isfunction
from typing import Tuple, List
from typing import Tuple, List


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, mult = 2, norm = True):
        super().__init__()


        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)
        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

class generator(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                #Upsample(dim_out) if not is_last else nn.Identity(),
                ConvNextBlock(dim_out, dim_out),
                ConvNextBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                ConvNextBlock(dim_out, dim_out),
                ConvNextBlock(dim_out, dim_out),
            ]))

        self.compress = nn.Conv2d(dims[-1],3,1)

    def forward(self, x):
        for convnext, convnext2, attn, convnext4, convnext5, attn2, convnext6, convnext7 in self.downs:
            x = convnext(x)
            x = convnext2(x)
            x = attn(x)
            # x = upsample(x)
            x = convnext4(x)
            x = convnext5(x)
            x = attn2(x)
            x = convnext6(x)
            x = convnext7(x)

        return self.compress(x)

class discriminator_v4(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity(),
                ConvNextBlock(dim_out, dim_out, norm=ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
            ]))

        self.compress = nn.Conv2d(dims[-1],1,1)
        # self.out = nn.Linear(256, 1)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        for convnext, convnext2, attn, downsample, convnect3, convnext4, attn2 in self.downs:
            x = convnext(x)
            x = convnext2(x)
            x = attn(x)
            x = downsample(x)
            x = convnect3(x)
            x = convnext4(x)
            x = attn2(x)

        x = self.compress(x)
        x = torch.flatten(x, start_dim=1)
        return self.out(x)
        
class discriminator_v3(nn.Module):
    def __init__(
        self,
        image_size,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
    ):
        super().__init__()
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        self.compress = nn.Conv2d(dims[-1],1,1)

        # last_size = (image_size // (2 ** (len(dim_mults) - 1)))**2
        # self.out = nn.Linear(last_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for convnext, convnext2, downsample in self.downs:
            x = convnext(x)
            x = convnext2(x)
            x = downsample(x)
        x = self.compress(x)
        x = torch.flatten(x, start_dim=1)
        return self.sigmoid(x)
    
class discriminator_v3_withLogits(nn.Module):
    def __init__(
        self,
        image_size,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
    ):
        super().__init__()
        if isinstance(dim_mults, list):
            dim_mults = tuple(dim_mults)
        assert isinstance(dim_mults, tuple), "Error: dim_mults is not a tuple"
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        self.compress = nn.Conv2d(dims[-1],1,1)

    def forward(self, x):
        for convnext, convnext2, downsample in self.downs:
            x = convnext(x)
            x = convnext2(x)
            x = downsample(x)
        x = self.compress(x)
        x = torch.flatten(x, start_dim=1)
        return x
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
    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d,):
        super().__init__()
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
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d, gp_coef=0.001):
        super().__init__()
        self.n_layers = n_layers
        self.gp_coef = gp_coef
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
    
    def make_r1_gp(self, discr_real_pred, real_batch):
        if torch.is_grad_enabled():
            grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
        else:
            grad_penalty = 0
        real_batch.requires_grad = False

        return grad_penalty * self.gp_coef


if __name__ == "__main__":
    # d = discriminator_v2("cuda:0" , 3,64)
    # # d = discriminator_v3(image_size=256, dim=32,    # 16 ==> 128
    # #     dim_mults=(8, 8, 4, 4, 2, 2, 2, 1, 1),     # 16 ==> 128
    # #     channels=3).to("cuda:1")

    # # input = torch.rand((8,3,256,256)).to("cuda:1")
    # # output = d(input)

    # # print(output.shape)
    # # print(output)     # 128:[1, 84] 256:[1,336]
    
    # from torchinfo import summary
    # device="cuda:0"
    # model = model = NLayerDiscriminator(3).to(device)
    # batch_size = 16
    # input_size = (batch_size, 3, 128, 128)
    # input = torch.randn(input_size).to(device)
    # discr_pred, discr_feature = model(input)
    # print(discr_pred[0].shape, discr_feature[0].shape, len(discr_pred), len(discr_feature))
    # #summary(model, input_data=input)
    
    # from torchinfo import summary
    device="cuda:0"
    model = model = NLayerDiscriminator(3).to(device)
    batch_size = 16
    input_size = (batch_size, 3, 256, 256)
    input = torch.randn(input_size).to(device)
    discr_pred, discr_feature = model(input)
    print(discr_pred.shape, discr_feature[0].shape, len(discr_pred), len(discr_feature))
    #summary(model, input_data=input)
