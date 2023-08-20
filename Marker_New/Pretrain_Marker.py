import Workspace
import os, cv2, glob
import numpy as np
import argparse
import Tiny_Imagenet
from omegaconf import OmegaConf
from copy import deepcopy
import random
import torch
import torch.nn as nn
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
import utils_model

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#from torchvision import transforms as transforms
import torch.nn.functional as F
import tqdm
print("import OK")

########################################################################
# def Noise(x,i):
#     i = i % 10
#     if i<=1:
#         x = T1
#     elif 1 < i and i <=3:
#         x = T2
#     elif 3 < i and i<=5:
#         x = T3
#     else:
#         x = x
#     return x


# define parser
def parse():
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(description='Pre train Marker Net')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate for training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, bigger help to converge, little faster')
    parser.add_argument('--input_channel', type=int, default=4, help='this is the channel of input images')
    parser.add_argument('--Channel_mid', type=int,default=320,help='Block numbers of mid layer')
    parser.add_argument("--num_workers", type=int, default=4, help='number of cpu to process data into dataloader')
    parser.add_argument('--num_blocks', default= 3, help="Decoder's block number")
    parser.add_argument('--num_bits',default= 5, help='The length of code')

    parser.add_argument("--ldm_config", type=str, default="sd/stable-diffusion-v-1-4-original/v1-inference.yaml",
       help="Path to the configuration file for the LDM model")
    parser.add_argument("--ldm_ckpt", type=str, default="sd/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt",
                        help="Path to the checkpoint file for the LDM model")

    parser.add_argument('--eval_epochs', type = int, default=10, help='eval epochs')
    return parser.parse_args()

########################################################################
# run the mode, train and test
def run(args,layernum=3):
    # import pdb; pdb.set_trace()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_dir = './data_pretrain/tiny-imagenet-200/'
    dataset_train = Tiny_Imagenet.TinyImageNet(data_dir, train=True)
    print("---The data is load in dataset")
    # print(dataset_train.keys())  # dict_keys(['boxes', 'images', 'labels'])
    dataloader = DataLoader(dataset_train, num_workers=8, batch_size=args.batch_size, shuffle=False)
    print("---It has been processed in dataloader")


    # define MarkerNet for Pre-training
    c_in = 4
    if c_in != args.input_channel:
        raise ValueError("Channel may be wrong! It's not 4!")
    code = ''.join([random.choice(['0', '1']) for _ in range(args.num_bits)])
    if layernum == 8:
        # the muli number, please refer: https://pic1.zhimg.com/v2-7476bc68a913afd13a2e4483a8869a04_r.jpg
        MarkerNet = Workspace.Marker_pretrain_Net_8(Channel_in=c_in, Channel_mid=args.Channel_mid,  Adj_Cr_channels_Low=c_in).to(device)
        DecoderNet_F1 = Workspace.Decoder_Net(args.num_blocks, args.num_bits, args.Channel_mid)
        DecoderNet_F2 = Workspace.Decoder_Net(args.num_blocks, args.num_bits, args.Channel_mid)
        DecoderNet_F3 = Workspace.Decoder_Net(args.num_blocks, args.num_bits, args.Channel_mid*2)
        DecoderNet_F4 = Workspace.Decoder_Net(args.num_blocks, args.num_bits, args.Channel_mid*4)
        DecoderNet_Feature_List = nn.ModuleList([DecoderNet_F1,DecoderNet_F2,DecoderNet_F3,DecoderNet_F4])

    elif layernum == 5:
        MarkerNet = Workspace.Marker_pretrain_Net_5(Channel_in=c_in, Channel_mid=args.Channel_mid,
                                                    Adj_Cr_channels_Low=c_in).to(device)
        DecoderNet_F1 = Workspace.Decoder_Net(args.num_blocks, args.num_bits, args.Channel_mid)
        DecoderNet_F2 = Workspace.Decoder_Net(args.num_blocks, args.num_bits, args.Channel_mid*2)
        DecoderNet_F3 = Workspace.Decoder_Net(args.num_blocks, args.num_bits, args.Channel_mid*4)
        DecoderNet_Feature_List = nn.ModuleList([DecoderNet_F1,DecoderNet_F2,DecoderNet_F3])

    elif layernum == 3:
        MarkerNet = Workspace.Marker_pretrain_Net_3(Channel_in=c_in, Channel_mid=args.Channel_mid,
                                                    Adj_Cr_channels_Low=c_in).to(device)
        DecoderNet_F1 = Workspace.Decoder_Net(args.num_blocks, args.num_bits, args.Channel_mid*2)
        DecoderNet_F2 = Workspace.Decoder_Net(args.num_blocks, args.num_bits, args.Channel_mid*4)
        DecoderNet_Feature_List = nn.ModuleList([DecoderNet_F1,DecoderNet_F2])
    else:
        assert ("Wrong layer number of CrossLowR, should be one of 8,5,3")
    print("---The network Marker is finished: ")#, MarkerNet, DecoderNet_Feature_List)

    # define DecoderNet
    DecoderNet_I = Workspace.Decoder_Net(args.num_blocks * 2, args.num_bits, c_in)
    print("---The Detector is finished")

    # define optimizer, use adam
    optimizer = torch.optim.Adam([
        {'params': MarkerNet.parameters(), 'lr': 0.001, },
        {'params': DecoderNet_Feature_List.parameters(), 'lr': 0.001,},
        {'params': DecoderNet_I.parameters(), 'lr': 0.001,}
    ])
    print("---Opt is done")

    # VAE Encoder part
    print(f'>>> Building LDM model with config {args.ldm_config} and weights from {args.ldm_ckpt}...')
    config = OmegaConf.load(f"{args.ldm_config}")
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, args.ldm_ckpt)     # all para
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model    # encoder para
    ldm_ae.eval()
    for param in [*ldm_ae.parameters()]:
        param.requires_grad = False

    print("---We start to train", len(dataloader))
    if args.mode == 'train':
        # train the model
        iterations = 0
        running_loss = 0.0
        MarkerNet.train()
        # print("Yep")
        for epoch in range(args.num_epochs):
            print("---We are training and in epoch: ", epoch)
            for i, data in tqdm.tqdm(enumerate(dataloader, 0),desc='Processing'):
                # print("Yep")
                # import pdb; pdb.set_trace()
                img_in = ldm_ae.encoder(data)
                code = ''.join([random.choice(['0', '1']) for _ in range(args.num_bits)])
                optimizer.zero_grad()
                mask, x_str = MarkerNet.forward(img_in,code)
                loss = 0
                loss += MarkerNet.CodeAcc_feature_loss(x_str,code,DecoderNet_Feature_List)
                loss += MarkerNet.recon_loss(x_str)
                loss += MarkerNet.Mask_loss(mask)
                with torch.no_grad():
                    x = x_str[-1]
                    x = ldm_ae.decode(x) # todo: find the VAE Decoder
                #X_n = Noise(x, i)
                loss += MarkerNet.CodeAcc_loss(x, code, DecoderNet_I)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                iterations = iterations + 1
                if iterations % args.print_every_iters == 0:  # print every xxx iters
                    print('Epoch %d iter: %d loss: %.5f' % (epoch + 1, iterations, running_loss*1000 / args.print_every_iters))
                    running_loss = 0

            # save model and test on the validation set
            torch.save(MarkerNet, 'Markernet.pt')
            torch.save(DecoderNet_Feature_List, 'DecoderNet_Feature_List.pt')
            torch.save(DecoderNet_I, 'DecoderNet_I.pt')

            # if (epoch+1) % args.eval_epochs == 0:
            #     # validation
            #     print("begin validation:")
            #     MarkerNet.eval()
            #     with torch.no_grad():
            #         x = x_str[-1]
            #         x = ldm_ae.decode(x)
            #         acc = MarkerNet.Acc_Code_eval(x,code,DecoderNet_I)
            #     print("Current epoch: ",epoch, "Bit Acc: ", acc)
            #     MarkerNet.train()



if __name__=='__main__':
    args = parse()
    print(args)
    run(args)