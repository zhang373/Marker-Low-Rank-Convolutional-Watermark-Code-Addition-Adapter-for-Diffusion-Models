import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import ldm
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from plms_Marker import PLMSSampler
import Workspace
import Pretrain_Marker
import random
import deeplake
import lpips

def load_model_from_config(config, ckpt, verbose=False):
    #送模型和模型数据进来
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model
def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--numlayer",
        type=int,
        default= 3,
        help="the number of layer of Marker net"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        # 这个是一个开关型的参数，只要被调用了，就会是true
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = set_parser()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = Pretrain_Marker.parse()
    DecoderNet_I = Workspace.Decoder_Net(args.num_blocks * 2, args.num_bits, args.Channel_in).to(device)
    c_in = 3
    if opt.numlayer == 3:
        MarkerNet = Workspace.Marker_pretrain_Net_3(Channel_in=c_in, Channel_mid=args.Channel_mid,
                                                    Adj_Cr_channels_Low=c_in).to(device)
    if opt.numlayer == 5:
        MarkerNet = Workspace.Marker_pretrain_Net_5(Channel_in=c_in, Channel_mid=args.Channel_mid,
                                                    Adj_Cr_channels_Low=c_in).to(device)
    if opt.numlayer == 8:
        MarkerNet = Workspace.Marker_pretrain_Net_8(Channel_in=c_in, Channel_mid=args.Channel_mid,
                                                    Adj_Cr_channels_Low=c_in).to(device)
    else:
        assert "Wrong layernum! should be 3/ 5/ 8"

    MarkerNet = torch.load('MarkerNet.pt')
    DecoderNet_I = torch.load('DecoderNet_I.pt')



    # 加参数进去
    config_marker = OmegaConf.load(
        "Marker_New.marker_finetune.yaml")
    model_Marker = load_model_from_config(config_marker, "models/ldm/text2img-large/model.ckpt")  # TODO: check path
    model_Marker = model_Marker.to(device)

    config = OmegaConf.load(
        "configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path
    model = model.to(device)

    #送上GPU

    if opt.plms:
        sampler = PLMSSampler(model, IsMarker=False)
        sampler_Marker = PLMSSampler(model_Marker, IsMarker=True, Marker=MarkerNet)
    else:
        # todo: this part cannot work! I have not changed the code here!
        sampler = DDIMSampler(model)
        sampler_Marker = DDIMSampler(model_Marker)
    #采样器

optimizer = torch.optim.Adam([
    {'params': MarkerNet.parameters(), 'lr': 0.001, }
])

ds = deeplake.load("hub://activeloop/coco-train")
dataloader = ds.pytorch(num_workers=0, batch_size=1, shuffle=False)
for epoch in range(opt.num_epochs):
    for i, data in tqdm.tqdm(enumerate(dataloader, 0), desc='Processing'):
        prompt = data.utf8_strings
        code = ''.join([random.choice(['0', '1']) for _ in range(args.num_bits)])
        optimizer.zero_grad()
        with model.ema_scope():
            # 这个是一个自己写的函数，我看看去
            uc = None   # 这个uc指的是uncondational的内部分
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])   #opt.n_samples是总体需要产生的采样数量
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])    #使用CLIP进行编码
                shape = [4, opt.H//8, opt.W//8]
                with torch.no_grad:
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta)
                samples_ddim_Marker, _ = sampler_Marker.sample(S=opt.ddim_steps,
                                                 conditioning=c, code=code,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta, Marker=MarkerNet)


                # todo: 我觉得不一定要解码，直接在feature map上操作应该就行; 不行,得用Decoder_I去进行解码
                # 上边这个完成了采样和生成
                with torch.no_grad:
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                    x_samples_ddim_Marker = model_Marker.decode_first_stage(samples_ddim_Marker)
                    x_samples_ddim_Marker = torch.clamp((x_samples_ddim_Marker + 1.0) / 2.0, min=0.0, max=1.0)

                # 这个完成了最后一步的生成，x_sample_ddim的数量是batch的数量
                    x_samples_ddim[0] = 255. * rearrange(x_samples_ddim[0], 'c h w -> h w c')
                    x_samples_ddim_Marker[0] = 255. * rearrange(x_samples_ddim_Marker[0], 'c h w -> h w c')

        with torch.no_grad:
            # additionally, save as grid
            grid = torch.stack([x_samples_ddim], 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=opt.n_samples)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c')
            Image_old = Image.fromarray(grid.astype(np.uint8))

            grid_marker = torch.stack([x_samples_ddim_Marker], 0)
            grid_marker = rearrange(grid_marker, 'n b c h w -> (n b) c h w')
            grid_marker = make_grid(grid_marker, nrow=opt.n_samples)

            # to image
            grid_marker = 255. * rearrange(grid_marker, 'c h w -> h w c')
            Image_Marker = Image.fromarray(grid_marker.astype(np.uint8))

        loss = 0
        code_= DecoderNet_I(Image_Marker)
        BCE_loss_fn = torch.nn.BCELoss()
        loss += BCE_loss_fn(code,code_)

        loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
        loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

        loss += loss_fn_alex(Image_old, Image_Marker)
        loss += loss_fn_vgg(Image_old, Image_Marker)

        loss.backward()
        optimizer.step()
