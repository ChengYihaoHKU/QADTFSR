import torch
import torch.nn.functional as F
import numpy as np

from omegaconf import OmegaConf
import wandb
from loguru import logger
from basicsr.data import degradations as degradations
from torch import nn
import lpips

from utils import util_image

from pathlib import Path
from torchvision.utils import save_image, make_grid
from utils.models import weights_init, Encoder, Decoder, Discriminator
from utils.discriminator import NLayerDiscriminator
import os, random, sys
from datapipe.datasets import create_dataset

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
import pyiqa
import torch.utils.data as udata 
import torch.multiprocessing as mp
import torch.distributed as dist
import datetime 
import torch.cuda.amp as amp
import math
class TrainerBase:
    def __init__(self, configs):
        self.configs = configs
        
        #  setup distributed training: self.num_gpus, self.rank
        self.setup_dist()
        
        # setup seed
        self.setup_seed()
        self.psnrmax = 0
        self.lpipsmax = 1
        self.current_iters = 0
    
    def setup_dist(self):
        # num_gpus = torch.cuda.device_count()
        num_gpus = 1
        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                timeout=datetime.timedelta(seconds=3600),
                backend='nccl',
                init_method='env://',
            )
        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0
    
    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith('.pth')
            save_dir = Path(self.configs.resmue).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)
        
        #  setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}
            
        #  text logger
        logtext_path = save_dir / 'training.log'
        if self.rank ==0:
            if logtext_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtext_path, format='{message}', mode='a', level='INFO')
            self.logger.add(sys.stdout, format='{message}')
        # wandb logger
        self.wandb_logging = self.configs.train.get('wandb_logging', False)
        if self.wandb_logging and self.rank == 0:
            wandb.init(
                project = 'DTLS_v2',
                name='lab 2 stage 1-6 1-1-0.01',
                dir = save_dir
            )
            self.wandb_logger = wandb
            
        #  checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()
        
        #  save iamges into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir
            
        #  logging the configs
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))
            if self.wandb_logging:
                self.wandb_logger.config.update(OmegaConf.to_container(self.configs, resolve=True))

    def build_model(self):
        
        # encoder
        self.encoder = Encoder()
        self.encoder.apply(weights_init)
        self.encoder = self.encoder.cuda()
        
        # decoder
        self.decoder = Decoder()
        self.decoder.apply(weights_init)
        self.decoder = self.decoder.cuda()
        
        # discriminator
        self.discriminator = NLayerDiscriminator(input_nc=3, gp_coef=self.configs.train.gp_coef)
        self.discriminator.apply(weights_init)
        self.discriminator = self.discriminator.cuda()
        
        # lpips metric
        if hasattr(self.configs, 'lpips'):
            lpips_net = self.configs.lpips.net
        else:
            lpips_net = 'vgg'
        if self.rank == 0:
            self.logger.info(f'LPIPS metric: {lpips_net}')
        self.lpips_loss = lpips.LPIPS(net=lpips_net).to(f'cuda:{self.rank}')
        for params in self.lpips_loss.parameters():
            params.requires_grad_(False)
        self.lpips_loss.eval()        
    
    def setup_optimization(self):
        self.optimizer_enc = torch.optim.AdamW(self.encoder.parameters(),
                                          lr=self.configs.train.lr, 
                                          weight_decay=self.configs.train.weight_decay)
        self.optimizer_dec = torch.optim.AdamW(self.decoder.parameters(),
                                          lr=self.configs.train.lr, 
                                          weight_decay=self.configs.train.weight_decay)
        self.optimizer_dis = torch.optim.AdamW(self.discriminator.parameters(),
                                          lr=self.configs.train.lr * 0.1, 
                                          weight_decay=self.configs.train.weight_decay)
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None
        self.BCE_loss = nn.BCEWithLogitsLoss()
        
        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler_enc = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer_enc,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )
            self.lr_scheduler_dec = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer_dec,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )
            self.lr_scheduler_dis = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer_dis,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )
           
    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler
    
    def resume_from_ckpt(self):
        self.iters_start = 0   
        if self.configs.train.ckpt_path:
            ckpt_path = Path(self.configs.train.ckpt_path)
            if ckpt_path.is_file():
                ckpt = torch.load(ckpt_path, map_location=f'cuda:0')
                self.encoder.load_state_dict(ckpt['enc'])
                self.decoder.load_state_dict(ckpt['dec'])
                self.discriminator.load_state_dict(ckpt['dis'])
                self.current_iters = ckpt['iters_start']
                self.iters_start = self.current_iters
                if self.rank == 0:
                    self.logger.info(f'Resume from {ckpt_path}')
            else:
                raise FileNotFoundError(f'Checkpoint file {ckpt_path} not found.')
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b*10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b
    
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        if realesrgan is None:
            realesrgan = self.configs.data.get(phase, dict).type == 'realesrgan'
        if realesrgan and phase == 'train':
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            if not hasattr(self, 'use_sharpener'):
                self.use_sharpener = USMSharp().cuda()

            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]
           
            if isinstance(self.configs.degradation.sf, int):
                sf = self.configs.degradation.sf
            #  uneven scaling factor
            else:
                sf_list = self.configs.degradation.sf
                weights = self.configs.degradation.weights
                sf = random.choices(sf_list, weights=weights, k=1)[0]
            # else:
            #     assert len(self.configs.degradation.sf) == 2
            #     sf = random.uniform(*self.configs.degradation.sf)

            if self.configs.degradation.use_sharp:
                im_gt = self.use_sharpener(im_gt)

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.configs.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.configs.degradation['gray_noise_prob']
            if random.random() < self.configs.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if random.random() < self.configs.degradation['second_order_prob']:
                # blur
                if random.random() < self.configs.degradation['second_blur_prob']:
                    out = filter2D(out, kernel2)
                # random resize
                updown_type = random.choices(
                        ['up', 'down', 'keep'],
                        self.configs.degradation['resize_prob2'],
                        )[0]
                if updown_type == 'up':
                    scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
                elif updown_type == 'down':
                    scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                        mode=mode,
                        )
                # add noise
                gray_noise_prob = self.configs.degradation['gray_noise_prob2']
                if random.random() < self.configs.degradation['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.configs.degradation['noise_range2'],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                        )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.configs.degradation['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                        )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)

            # resize back
            if self.configs.degradation.resize_back:
                out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')

            # clamp and round
            im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
            self.lq, self.gt, flag_nan = replace_nan_in_batch(im_lq, im_gt)
            if flag_nan:
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return {'lq':self.lq, 'gt':self.gt}
        
        elif phase == 'val':
            offset = self.configs.train.get('val_resolution', 512)
            for key, value in data.items():
                h, w = value.shape[2:]
                if h > offset and w > offset:
                    h_end = int((h // offset) * offset)
                    w_end = int((w // offset) * offset)
                    data[key] = value[:, :, :h_end, :w_end]
                else:
                    h_pad = math.ceil(h / offset) * offset - h
                    w_pad = math.ceil(w / offset) * offset - w
                    padding_mode = self.configs.train.get('val_padding_mode', 'reflect')
                    data[key] = F.pad(value, pad=(0, w_pad, 0, h_pad), mode=padding_mode)
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        else:
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}

    def musiq_eval_init(self):
        musiq = pyiqa.create_metric('musiq', device=f'cuda:{self.rank}')
        self.musiq = musiq
            
    def train_step(self, data):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

        self.timestep = self.timestep.squeeze().cuda()
        latent_img, h = self.encoder(data['lq'], self.timestep)
        recon_img = self.decoder(latent_img, self.timestep, h)
        # discriminator
        data['gt'].requires_grad = True
        target_score,_ = self.discriminator(data['gt'])
        dis_err_real = self.BCE_loss(target_score, torch.ones_like(target_score))
        
        real_loss = dis_err_real
        real_loss.backward(retain_graph=True)
        
        current_score, _ = self.discriminator(recon_img.detach())
        dis_err_fake = self.BCE_loss(current_score, torch.zeros_like(current_score))
        dis_err_fake.backward()
        self.optimizer_dis.step()
        
        # encoder and decoder
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        
        err_mse = ((recon_img - data['gt']) ** 2).mean()
        err_lpips = self.lpips_loss(recon_img, data['gt']).mean()
        
        update_score, _ = self.discriminator(recon_img)
        err_dis = self.BCE_loss(update_score, torch.ones_like(update_score))
        


        (err_mse * self.configs.train.error_weight_1[0] + err_lpips * self.configs.train.error_weight_1[1] + err_dis * self.configs.train.error_weight_1[2]).backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()
        
        with torch.no_grad():
            score_lq2 = self.musiq((torch.clamp((recon_img.detach() + 1) / 2, min=0, max=1)).cuda())
            aug_2 = 0.5 * recon_img.detach() + 0.5 * data['lq']
            self.timestep2 = self.score_2_timestep(score_lq2)
            self.timestep2 = self.timestep2.squeeze().cuda()
        # make weak degradation on the second order
        
        latent_img2, h2 = self.encoder(aug_2, self.timestep2)
        recon_img2 = self.decoder(latent_img2, self.timestep2, h2)
        
        err_mse2 = ((recon_img2 - data['gt']) ** 2).mean()
        err_lpips2 = self.lpips_loss(recon_img2, data['gt']).mean()
        
        update_score2, _ = self.discriminator(recon_img2)
        err_dis2 = self.BCE_loss(update_score2, torch.ones_like(update_score2))


        ( err_mse2 * self.configs.train.error_weight_2[0] + err_lpips2 * self.configs.train.error_weight_2[1] + err_dis2 * self.configs.train.error_weight_2[2]).backward()

        self.optimizer_enc.step()
        self.optimizer_dec.step()
        
        
        
        if self.rank == 0 and self.current_iters % self.configs.train.log_freq[0] == 0:
            self.logger.info(f'[{self.current_iters}/{self.configs.train.iterations}] '
                                
                                f'err_mse: {err_mse.item():.4f} '
                                f'err_lpips: {err_lpips.item():.4f} '
                                f'err_dis: {err_dis.item():.4f} '
                                f'err_mse2: {err_mse2.item():.4f} '
                                f'err_lpips2: {err_lpips2.item():.4f} '
                                f'err_dis2: {err_dis2.item():.4f} '
                                                                    )
            
            if self.wandb_logging:
                self.wandb_logger.log({
                    'train/err_mse': err_mse.item(),
                    'train/err_lpips': err_lpips.item(),
                    'train/err_dis': (err_dis * 1e-2).item(),
                    'train/err_mse2': err_mse2.item(),
                    'train/err_lpips2': err_lpips2.item(),
                    'train/err_dis2': (err_dis2 * 1e-2).item(),
                }, step=self.current_iters)

        if self.local_logging and self.current_iters % self.configs.train.log_freq[1] == 0:
            combined = torch.cat([data['lq'], data['gt'], recon_img, aug_2, recon_img2], dim=0)
            save_image(combined, self.image_dir / 'train' / f'{self.current_iters:08d}_combined.png',
                        nrow=data['lq'].shape[0],  # Number of images per row (batch size)
                        normalize=True,
                        value_range=(-1, 1))
            if self.wandb_logging:
                combined = torch.cat([data['lq'], data['gt'], recon_img, aug_2, recon_img2], dim=0)
                self.wandb_logger.log({
                    'train/combined': wandb.Image(make_grid(combined, nrow=data['lq'].shape[0],)),
                }, step=self.current_iters)
    
    def validation(self, phase='val'):
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        batch_size = self.configs.train.batch[1]
        num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
        mean_psnr = 0.0
        mean_lpips = 0.0
        last_recon_img = None
        last_recon_img_2 = None
        
        for ii, data in enumerate(self.dataloaders[phase]):
            data = self.prepare_data(data, phase= 'val')
            if 'gt' in data:
                im_lq = data['lq']
                im_gt = data['gt']
            else:
                im_lq = data['lq']
            score_lq = self.musiq(torch.clamp(((im_lq+1)/2).cuda(), min=0, max=1))
            self.timestep = self.score_2_timestep(score_lq)
            self.timestep = self.timestep.squeeze().cuda()
            latent_img, h = self.encoder(im_lq, self.timestep)
            recon_img = self.decoder(latent_img, self.timestep, h)
            recon_img = torch.clamp(recon_img, min=-1, max=1)
            last_recon_img = recon_img
            score_lq2 = self.musiq((recon_img + 1) / 2).cuda()
            aug2 = recon_img * 0.5 + im_lq * 0.5

            self.timestep2 = self.score_2_timestep(score_lq2)
            self.timestep2 = self.timestep2.squeeze().cuda()
            
            latent_img2, h2 = self.encoder(aug2, self.timestep2)
            recon_img = self.decoder(latent_img2, self.timestep2, h2)
            recon_img = torch.clamp(recon_img, min=-1, max=1)
            last_recon_img_2 = recon_img
            
            if 'gt' in data:
                # calculate psnr and lpips
                mean_psnr += util_image.batch_PSNR(
                            recon_img * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                           
                            )
                mean_lpips += self.lpips_loss(
                            recon_img,
                            im_gt,
                            ).sum().item()
            if (ii + 1) % self.configs.train.log_freq[2] == 0:
                self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')
        if 'gt' in data:
            mean_psnr /= (len(self.dataloaders[phase]) * batch_size)
            mean_lpips /= (len(self.dataloaders[phase]) * batch_size)
            
            if self.psnrmax < mean_psnr:
                self.psnrmax = mean_psnr
                self.save_ckpt(metric='psnr')
                
            if self.lpipsmax > mean_lpips:
                self.lpipsmax = mean_lpips
                self.save_ckpt(metric='lpips')
                
                        
            self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}')
            if self.local_logging:
                combined = torch.cat([im_lq, im_gt, aug2, last_recon_img, last_recon_img_2], dim=0)
                save_image(
                    combined, self.image_dir / 'val' / f'{self.current_iters:08d}_combined.png', 
                      nrow=im_lq.shape[0],  # Number of images per row (batch size)
                      normalize=True, value_range=(-1, 1)
                )
            if self.wandb_logging:
                self.wandb_logger.log({
                    f'{phase}/psnr': mean_psnr,
                    f'{phase}/lpips': mean_lpips,
                }, step=self.current_iters)
                # log images using wandb
                combined = torch.cat([im_lq, im_gt, recon_img], dim=0)
                self.wandb_logger.log({
                    f'{phase}/combined': wandb.Image(make_grid(combined, nrow=im_lq.shape[0],)),
                }, step=self.current_iters)
        self.logger.info("="*100)
  
    def score_2_timestep(self, score):
        timestep = torch.clamp((80 - score) // 10, min=0)
        # Ensure it's a 1D tensor with shape [batch_size]
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)  # Make it 1D
        elif timestep.dim() > 1:
            # If it's more than 1D, flatten it to 1D
            timestep = timestep.view(-1)
        return timestep

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer_enc.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
            for params_group in self.optimizer_dec.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
            for params_group in self.optimizer_dis.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler_enc.step()
                self.lr_scheduler_dec.step()
                self.lr_scheduler_dis.step()
        
    def save_ckpt(self, metric='psnr'):
        if self.rank == 0:
            if metric not in ['psnr', 'lpips']:
                ckpt_path = self.ckpt_dir / 'model_{}.pth'.format(self.current_iters)
                ckpt = {
                        'iters_start': self.current_iters,
                        
                        'enc': self.encoder.state_dict(),
                        'dec': self.decoder.state_dict(),
                        'dis': self.discriminator.state_dict(),                    
                        }
                torch.save(ckpt, ckpt_path)
            else:
                ckpt_path = self.ckpt_dir / 'model_{}.pth'.format(metric)
                ckpt = {
                        'iters_start': self.current_iters,
                        
                        'enc': self.encoder.state_dict(),
                        'dec': self.decoder.state_dict(),
                        'dis': self.discriminator.state_dict(),                    
                        }
                torch.save(ckpt, ckpt_path)
            
    def train(self):
        self.init_logger()
        
        self.build_model()
        
        self.setup_optimization()
        
        self.resume_from_ckpt()
        
        self.build_dataloader()
        self.musiq_eval_init()
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters += 1
            
            data =self.prepare_data(next(self.dataloaders['train']))
            score_lq = self.musiq(((data['lq']+1)/2).cuda())
            self.timestep = self.score_2_timestep(score_lq)
            
            self.train_step(data)
            
            # validation phase
            if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 5000) == 0:
                with torch.no_grad():
                    self.validation()
                    self.save_ckpt(metric='Normal')
            
            self.adjust_lr()
            
          
class EvaluatorBase():
    def __init__(self, configs):
        self.configs = configs
        
    def build_model(self):
        # encoder
        self.encoder = Encoder()
        self.encoder.apply(weights_init)
        self.encoder = self.encoder.cuda(device=self.device)
        
        # decoder
        self.decoder = Decoder()
        self.decoder.apply(weights_init)
        self.decoder = self.decoder.cuda(device=self.device)
        
        # discriminator
        self.discriminator = Discriminator()
        self.discriminator.apply(weights_init)
        self.discriminator = self.discriminator.cuda(device=self.device)
        if self.configs.train.ckpt_path:
            assert os.path.exists(self.configs.train.ckpt_path), f'Checkpoint path {self.configs.train.ckpt_path} does not exist'
            
            ckpt = torch.load(self.configs.train.ckpt_path, map_location=self.device)
            self.encoder.load_state_dict(ckpt['enc'])
            self.decoder.load_state_dict(ckpt['dec'])
            # self.discriminator.load_state_dict(ckpt['dis'])
        else:
            raise ValueError('Checkpoint path is not specified')
    
    def build_dataloader(self):
        # make datasets
        datasets = {'eval': create_dataset(self.configs.data.get('eval', dict)), }

        # make dataloaders
        dataloaders = {'eval': udata.DataLoader(datasets['eval'],
                                              batch_size=self.configs.train.batch[1],
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              pin_memory=True,
                                             )}
        self.datasets = datasets
        self.dataloaders = dataloaders
    
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='eval'):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cuda(device=self.device).to(dtype=dtype)
                
    def score_2_timestep(self, score):
        timestep = torch.clamp((80 - score) // 10, min=0)
        return timestep
    
    def eval_step(self, data):
        with torch.no_grad():
            score_lq = self.musiq(torch.clamp(((data['lq']+1)/2), min=0, max=1))
            self.timestep = self.score_2_timestep(score_lq)
            self.timestep = self.timestep.squeeze().cuda(device=self.device)
            latent_img, h = self.encoder(data['lq'], self.timestep)
            recon_img_1 = self.decoder(latent_img, self.timestep, h)
            recon_img_1 = torch.clamp(recon_img_1, min=-1, max=1)
            # performance the second stage
            score_lq = self.musiq(torch.clamp(((recon_img_1+1)/2), min=0, max=1))
            self.timestep = self.score_2_timestep(score_lq)
            self.timestep = self.timestep.squeeze().cuda(device=self.device)
            aug_2 = 0.5 * data['lq'] + 0.5 * recon_img_1
            latent_img, h = self.encoder(aug_2, self.timestep)
            recon_img_2 = self.decoder(latent_img, self.timestep, h)
            recon_img_2 = torch.clamp(recon_img_2, min=-1, max=1)
            
        return recon_img_2
    
    def eval(self):
        self.device = 'cuda:0'
        self.build_model()
        self.build_dataloader()
        self.encoder.eval()
        self.decoder.eval()
        self.musiq = pyiqa.create_metric('musiq', device=self.device)
        if not os.path.exists(self.configs.save_dir):
            os.makedirs(self.configs.save_dir)
      
        for _, data in enumerate(self.dataloaders['eval']):
            self.prepare_data(data)
            recon_img = self.eval_step(data)
            # save image with batch size

            for jj in range(recon_img.shape[0]):
                img_path = os.path.join(self.configs.save_dir, os.path.basename(data['path'][jj]))
                save_image(recon_img[jj], img_path , normalize=True)          


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id) 

def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag


