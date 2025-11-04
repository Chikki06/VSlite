import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torch
from AffineT import *
from PIL import Image as imp
from IPython.core.debugger import set_trace
import spectral
import tifffile as tif
import scipy.ndimage
import cv2


class ImageDataset(Dataset):
    def __init__(self, root):

        self.filesI = os.path.join(root, 'ir')
        self.list_I = os.listdir(self.filesI)
        self.filesT = os.path.join(root, 'he')
        self.list_T = os.listdir(self.filesT)
        self.filesV = os.path.join(root, 'vis')
        self.list_V = os.listdir(self.filesV) 
        
#         random.shuffle(self.list_T)

        
    def __getitem__(self, index):

        I_name = os.path.join(self.filesI,self.list_I[index%len(self.list_I)])
        #img = Image.open(I_name)
#        img = Image.open(I_name)
#        img = im['xx']
        #im = sio.loadmat(I_name)
        img = np.load(I_name)
        img = np.transpose(img, (2, 0, 1))
        ir_data = img[:, :, :]
       # for i in range(ir_data.shape[0]-1):
        #    ir_data[i,:] = ir_data[i,:,:]/(ir_data[-1,:,:]+1e-8)
        #ir_data = ir_data[:-2,:,:]
        
        ir_data[ir_data>5] = 5
        ir_data[ir_data<0] = 0
        img = ir_data/1
        V_name = os.path.join(self.filesV,self.list_I[index%len(self.list_V)])
        #img = Image.open(I_name)
#        img_T = Image.open(T_name)
        #im = sio.loadmat(V_name)
        img_V = np.load(V_name)/256
        img_V = np.transpose(img_V, (2, 0, 1))
        #img_V[img_V>1] = 1
        #img_V[img_V<0] = 0
        #img_V = (img_V - .5)/.5
        
        H_name = os.path.join(self.filesT,self.list_T[index%len(self.list_T)])
        #img = Image.open(I_name)
#        img_T = Image.open(T_name)
        #im = sio.loadmat(H_name)
        img_H = np.load(H_name)/256
#         em = np.sum(img_H,2)==0
#         for ii in range(3):
#             img_H[em,ii] = 1
        #set_trace()
        img_H[img_H>1] = 1
        img_H[img_H<0] = 0
        img_H = (img_H - .5)/.5
        #temp = np.sum(img_T,axis=2)>2.7
       # mask_he = np.zeros((256,256,1))
        #mask_he[:,:,0] = 1 - temp
       # mask_he[:,:,1] = 1 - temp        

        
#        set_trace()

        lr_transforms = [  
#                transforms.Resize((256, 256), Image.BICUBIC),
#        transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
#                        transforms.Resize((128, 128), Image.BICUBIC),
                        transforms.ToTensor(),
#                        transforms.Normalize((.5,0.5,0.5), (0.5,0.5,0.5))
                        ]
    
        hr_transforms = [
#            transforms.Resize((256, 256), Image.BICUBIC),
#            transforms.Resize((256, 256), Image.BICUBIC),
       # transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1),
                        transforms.ToTensor(),
                     #   transforms.Normalize((.5,0.5,0.5), (0.5,0.5,0.5))
                        ]    
        
    
        
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        
        
        shear = random.uniform(-.05,0.05)
        translation = (random.uniform(-.15,0.15),random.uniform(-.15,0.15))
        rotation = random.uniform(0,180)
        scale = (random.uniform(.95,1.05),random.uniform(.95,1.05))
#        S = random.randint(16,33)
        affine_transform = Affine(rotation_range=rotation, translation_range=translation, shear_range = shear, zoom_range = scale)
#        
#        
        CRx = random.randint(0,127)
        CRy = random.randint(0,127)

        fact = 128
#        img = img[CRx:CRx+fact,CRy:CRy+fact,(0,1,2,4,5,6,7,8,10,11,12,16)]

        trans1 = transforms.ToTensor()
 

        target_size = (256, 256)

        img_upsampled = np.zeros((img.shape[0], target_size[1], target_size[0]), dtype=img.dtype)

        for channel in range(img.shape[0]):
            img_upsampled[channel, :, :] = cv2.resize(img[channel, :, :], target_size, interpolation=cv2.INTER_CUBIC)

        combined_data = np.concatenate([img_upsampled, img_V], axis=0)

        combined_data = (affine_transform(torch.from_numpy(combined_data).float()) - .5)/.5
       # mask_he = self.lr_transform(mask_he) 
        #mask_ir = self.lr_transform(mask_ir) 
        name = self.list_I[index%len(self.list_I)];
       # set_trace()
        #img_V = self.hr_transform(imp.fromarray(np.repeat(np.expand_dims(img_V, axis=2),3,axis = 2)))#.repeat(3, 1, 1)
        #set_trace()
        
        
        img_T = affine_transform(self.lr_transform(img_H)-1) + 1 
        #img_T = affine_transform(img_T[:,CRx:CRx+fact,CRy:CRy+fact]) + 1
        #img = affine_transform(img[(10,18,19,20,21,22),CRx:CRx+fact,CRy:CRy+fact]) - 1
      #  mask_he = affine_transform(mask_he[:,CRx:CRx+fact,CRy:CRy+fact])
       # mask_ir = affine_transform(mask_ir[:,CRx:CRx+fact,CRy:CRy+fact])
        #img = torch.cat((img,mask_ir),0)

        return {'input': combined_data, 'target': img_T, 'name': name}

    def __len__(self):
        return len(self.list_I)

    
class ImageDataset_biopsies(Dataset):
    def __init__(self, root):


        self.parent = root
        self.filesI = os.path.join(self.parent,"IR")
        self.list_I = os.listdir(self.filesI)

        image_dir = 'D:/VSlite/datasets/ImagesT'
        tif_files = [os.path.splitext(file)[0] for file in os.listdir(image_dir) if file.endswith('.tif')]
        self.list_I = [file for file in self.list_I if file not in tif_files]
        #self.list_I = [item for item in os.listdir(self.filesI) if "5419-4" in item ]
        
    def __getitem__(self, index):

        FILE = os.listdir(self.parent + '/IR/' + self.list_I[index] +'/')
        print(self.list_I[index] )
        hdr_file = [f for f in FILE if f.endswith('.hdr')][0]
        bsq_file = hdr_file.replace(".hdr", ".bsq")
       
        name = self.parent + '/IR/' + self.list_I[index]  +'/'
        image = spectral.envi.open(name+hdr_file, name+bsq_file)
        img = np.array(image.load())
        img = np.transpose(img, (2, 0, 1))

        #img = Image.open(I_name)
#        img = Image.open(I_name)
#        img = im['xx']
 
        img = img[:,:,:].copy()
       # temp = np.sum(img[:,:,:],2)<.2
       # mask_ir = np.zeros((256,256,1))
        #mask_ir[:,:,0] = 1 - temp
        #mask_ir[:,:,1] = 1 - temp
        #set_trace()
        img[img>5] = 5
        img[img<-5] = -5
        img = img/1
        


       # V_name = os.path.join(self.filesV,self.list_I[index%len(self.list_V)])
        #img = Image.open(I_name)
#        img_T = Image.open(T_name)
       # im = sio.loadmat(V_name)
        name = self.parent + '/VIS/' + self.list_I[index]  +'_VIS_REG.tif'
        img_V = tif.imread(name)/256
        img_V = img_V.astype(np.float32)
        img_V = np.transpose(img_V, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img_V = torch.from_numpy(img_V).float()
        img_V[img_V>1] = 1
        img_V[img_V<0] = 0
   

   

        
#        set_trace()

        lr_transforms = [  
#                transforms.Resize((RS, RS), Image.BICUBIC),
#        transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5),
#                        transforms.Resize((128, 128), Image.BICUBIC),
                        transforms.ToTensor(),
#                        transforms.Normalize((.5,0.5,0.5), (0.5,0.5,0.5))
                        ]
    
        hr_transforms = [
#            transforms.Resize((128, 128), Image.BICUBIC),
#            transforms.Resize((256, 256), Image.BICUBIC),
       # transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=-.15),
                        transforms.ToTensor(),
                       # transforms.Normalize((.5,0.5,0.5), (0.5,0.5,0.5))
                        ]    
        
    
        

#        img = img[CRx:CRx+fact,CRy:CRy+fact,(0,1,2,4,5,6,7,8,10,11,12,16)]

        trans1 = transforms.ToTensor()
       # mask_he = self.lr_transform(mask_he) 
        #mask_ir = self.lr_transform(mask_ir) 
        name = self.list_I[index%len(self.list_I)];



        return {'ir': img[:,:,:],'vis': img_V, 'name': name}

    def __len__(self):
        return len(self.list_I)

