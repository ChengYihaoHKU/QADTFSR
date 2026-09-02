import math
import torch
from torch import nn
from einops import rearrange
from inspect import isfunction

from torch.ao.nn.qat import Conv2d


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# def Upsample(dim):
#     return nn.ConvTranspose2d(dim, dim, 4, 2, 1)
#
# def Downsample(dim):
#     return nn.Conv2d(dim, dim, 4, 2, 1)

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
        # self.norm = nn.BatchNorm2d(dim)
        # self.norm = nn.GroupNorm(dim // 32, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)
        # self.norm = nn.BatchNorm2d(dim)
        # self.norm = nn.GroupNorm(dim // 32, dim)

    def forward(self, x):
        x = self.fn(x)
        return self.norm(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim*2)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, 1, 1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            weight, bias = torch.split(condition, x.shape[1],dim=1)
            h = h * (1 + weight) + bias

        h = self.net(h)
        return h + self.res_conv(x)


class ConvNextBlock_dis(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim*2)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            nn.BatchNorm2d(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, 1, 1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            weight, bias = torch.split(condition, x.shape[1],dim=1)
            h = h * (1 + weight) + bias

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


class NoiseInjection(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, 1))

    def forward(self, feat, t):
        batch, _, height, width = feat.shape
        noise = torch.randn(batch, 1, height, width).to(feat.device)
        t = self.mlp(t)
        return feat + t.mean() * noise


# model
class Encoder(nn.Module):
    def __init__(
        self,
        dim = 32,
        dim_mults=(1, 2, 4, 4, 8, 8, 16, 32),
        channels = 3,
        with_time_emb = True,
    ):
        super().__init__()
        self.channels = dim

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )


        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        self.initial = nn.Conv2d(channels, dim, 1)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                nn.AvgPool2d(2),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if ind >= (num_resolutions - 3) else nn.Identity(),
                ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim),
            ]))

        # self.soft = nn.Softmax2d()

    def forward(self, x, time):
        x = self.initial(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # print(t[0])
        h = []
        for convnext, downsample, attn, convnext2 in self.downs:
            x = convnext(x, t)
            x = downsample(x)
            h.append(x)
            x = attn(x)
            x = convnext2(x, t)
        return x, h


class Decoder(nn.Module):
    def __init__(
        self,
        dim = 32,
        out_dim = 3,
        dim_mults=(1, 2, 4, 4, 8, 8, 16, 32),
        channels = 3,
        with_time_emb = True,
    ):
        super().__init__()
        self.channels = channels

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= num_resolutions
            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                nn.Upsample(scale_factor=2, mode='nearest'),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if ind < 3 else nn.Identity(),
                ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim),
                # NoiseInjection(time_dim) if not is_last else nn.Identity(),
            ]))

        # self.final_conv = ConvNextBlock(dim, 3, time_emb_dim=time_dim)
        # self.final_conv = nn.Sequential(nn.Conv2d(dim, 3, 1))
        self.final_conv = nn.Conv2d(dim, 3, 1)

    def forward(self, x, time, h):
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        for convnext, upsample,  attn, convnext2 in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = upsample(x)
            x = attn(x)
            x = convnext2(x, t)
            # x = noise(x, t)
        return self.final_conv(x)

# TODO patch gan discriminator
class Discriminator_patchgan(nn.Module):
    def __init__():
        pass

class Discriminator(nn.Module):
    def __init__(
            self,
            dim=32,
            dim_mults=(1, 2, 4, 4, 8, 8, 16, 32),
            channels=3,
            with_time_emb=True,
    ):
        super().__init__()
        self.channels = dim

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        self.initial = nn.Conv2d(channels, dim, 1)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock_dis(dim_in, dim_out, norm=ind != 0),
                nn.AvgPool2d(2),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))) if ind >= (num_resolutions - 3) and not is_last else nn.Identity(),
                ConvNextBlock_dis(dim_out, dim_out),
            ]))
        dim_out = dim_mults[-1] * dim

        self.out = nn.Conv2d(dim_out, 1, 1, bias=False)


    def forward(self, x):
        x = self.initial(x)
        for convnext, downsample, convnext2 in self.downs:
            x = convnext(x)
            x = downsample(x)
            # x = attn(x)
            x = convnext2(x)
        return self.out(x).view(x.shape[0], -1)


if __name__ == "__main__":
    model = Encoder().to("cuda:1")
    model_2 = Decoder().to("cuda:1")

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for param in model_2.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    for buffer in model_2.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))