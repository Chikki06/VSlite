import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import scipy.io
import torchvision.transforms as transforms
from torchvision.utils import save_image
#from sync_batchnorm import convert_model
from PIL import Image
import cv2
import tifffile

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import time

#from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch
from DRB import *

from datasets_testcrop import *
from IPython.core.debugger import set_trace





class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,1), targets)


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=8500, help='number of epochs of training')
    parser.add_argument('--dataset_name', type=str, default="facades", help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.599, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=1, help='epoch from which to start lr decay')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threaDSA to use during batch generation')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=3000, help='interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
    opt = parser.parse_args()
    print(opt)

#    writer = SummaryWriter()
    os.makedirs('imagesT/%s' % opt.dataset_name, exist_ok=True)
    os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)
    
    cuda = True if torch.cuda.is_available() else False
    

#    criterion_Class = IW_MaxSquareloss(ignore_index= 0, num_class=11, ratio=0.2) 
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height//2**4, opt.img_width//2**4)
    

    
    netG = AttGenerator_ResNet50_SkipCon()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
 # model = nn.DataParallel(model)


        netG.load_state_dict(torch.load('saved_models/15_net_G.pth'))
        netG = nn.DataParallel(netG)
        netG.to(device)
        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()
        netG = netG.eval() 


        print("Let's use", torch.cuda.device_count(), "GPUs!")

   

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor2 = torch.FloatTensor
    CK = 100
    temploss = 1
    count = 0
    Step = 1
    prev_time = time.time()
    loss_D = Tensor(1)   
    prev_time = time.time()
    loss_DA = Tensor(1)
    loss_DB = Tensor(1)
    loss_DSA = Tensor(1)
    loss_DSB = Tensor(1)
    BATCH_SIZE = 2
    dataloader = DataLoader(ImageDataset_biopsies("../../../Sections/"),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    for i, batch in enumerate(dataloader):
        
        total_time = 0
        num_iterations = 0
        ir = ((batch['ir'].type(Tensor2))-0)/1
        vis = ((batch['vis'].type(Tensor2))-0)/1
        name = (batch['name'][0])

        fact = 1024
        scaling_factor = (0.69/2)
        _,bands,_,_ = ir.shape
        b,_,_,_ = ir.shape
        _,_,r,c = ir.shape
        r=int(r/scaling_factor+1)
        c=int(c/scaling_factor+1)
        rn = np.ceil(r/fact)
        cn = np.ceil(c/fact)
        DATA = torch.zeros(BATCH_SIZE, 13, fact, fact)  # Change based on your channel dimension
        
        fhr = torch.zeros(b, 3, int(rn)*fact+1, int(cn)*fact+1).data.cpu().numpy()

        if count>=0:
            for kk in range(b):
                patch_count = 0
                patches = []
                I = range(0, int(rn)*fact+1, fact)
                J = range(0, int(cn)*fact+1, fact)
                for ii in range(len(I)-1):
                    for jj in range(len(J) - 1):
                        xx = ir[:, :, int(I[ii] * scaling_factor):int(I[ii+1] * scaling_factor), int(J[jj] * scaling_factor):int(J[jj+1] * scaling_factor)]
                        pad_rows = int(fact * scaling_factor) - xx.shape[2] if xx.shape[2] < int(fact * scaling_factor) else 0
                        pad_cols = int(fact * scaling_factor) - xx.shape[3] if xx.shape[3] < int(fact * scaling_factor) else 0

                        xx = F.pad(xx, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                        xx = F.interpolate(xx, size=(fact, fact), mode='bicubic', align_corners=True)
                        vv = vis[:, :, int(I[ii]):int(I[ii+1]), int(J[jj]):int(J[jj+1])]
                        pad_rows = fact - vv.shape[2] if vv.shape[2] < fact else 0
                        pad_cols = fact - vv.shape[3] if vv.shape[3] < fact else 0

                        vv = F.pad(vv, (0, pad_cols, 0, pad_rows), mode='constant', value=0)

                        DATA[patch_count, :bands, :, :] = xx
                        DATA[patch_count, bands:, :, :] = vv
                        patches.append((ii, jj))
                        patch_count += 1

                        if patch_count == BATCH_SIZE:  # Process the batch
                            start_time = time.time()
                            
                            output = netG(DATA[:patch_count].type(Tensor)).squeeze().data.cpu().numpy() * 0.5 + 0.5
                            total_time += (time.time() - start_time)
                            num_iterations += 1
                            #set_trace()
                            
                            for idx, (pi, pj) in enumerate(patches):
                                
                                fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]

                            patch_count = 0
                            patches = []

                if patch_count > 0:  # Process remaining patches
                    output = netG(DATA[:patch_count].type(Tensor)).squeeze().data.cpu().numpy() * 0.5 + 0.5
                    for idx, (pi, pj) in enumerate(patches):
                        fhr[kk, :, int(I[pi]):int(I[pi+1]), int(J[pj]):int(J[pj+1])] = output[idx]




#         aa = 'imagesT/'+name+'.png'
#         #set_trace()
#         fhr = np.transpose(fhr, (0, 2,3,1))
#         fhr = Image.fromarray(np.uint8(fhr[0,:r,:c,:]*256))
#         fhr.save(aa)
#         he = 0
#         fhr = 0
        average_time_per_operation = total_time / num_iterations if num_iterations > 0 else 0
        print(f"Average time per operation: {average_time_per_operation} seconds")
        fhr = np.uint8(np.transpose(fhr, (0, 2,3,1))*255)
        #set_trace()
        fhr = fhr[0,:r,:c,:]
        aa = 'imagesT/'+name+'.tif'
        pyramid_levels = [fhr]
        for i in range(1, 4): 
            scale = 2 ** i
            resized = cv2.resize(fhr, (fhr.shape[1] // scale, fhr.shape[0] // scale))
            pyramid_levels.append(resized)

        with tifffile.TiffWriter(aa, bigtiff=True) as tiff:
            for level in pyramid_levels:
                tiff.write(level, photometric='rgb', compression='jpeg', tile=(256, 256), metadata={'axes': 'YXS'})

         #   if epoch==299 or epoch==499 or epoch==699:
                # Save model checkpoints
    #                torch.save(netG_A2B.module.state_dict(), 'saved_models/netG_A2B%d.pth' % batches_done)
    #                torch.save(discriminator.state_dict(), 'saved_models/%s/discriminator_%d.pth' % batches_done)
    #                torch.save(netG_B2A.module.state_dict(), 'saved_models/netGB2A%d.pth' % batches_done)
    #                
if __name__ == '__main__':
    
    main()
