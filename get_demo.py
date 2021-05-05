import sys
import torch
import torch.nn as nn
import numpy as np
from model.pose_generator_norm import Generator
from dataset.lisa_dataset_test import DanceDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import numpy as np
import math
import itertools
import time
import datetime

from matplotlib import pyplot as plt
#import cv2
from dataset.output_helper import save_2_batch_images
import argparse
from scipy.io.wavfile import write

parser = argparse.ArgumentParser()

parser.add_argument(
        "--model",
        default="./pretrain_model/generator_0400.pth",
        metavar="FILE",
        help="path to pth file",
        type=str
    )

parser.add_argument(
        "--data",
        default="./dataset/lisa_revised_pose_pairs.json",
        metavar="FILE",
        help="path to pth file",
        type=str
    )

parser.add_argument("--count", type=int, default=100)
parser.add_argument(
        "--output",
        default="/data0/htang/projects/Music-Dance-Video-Synthesis/Demo/test",
        metavar="FILE",
        help="path to output",
        type=str
    )
parser.add_argument(
        "--keypoint",
        default="/data0/htang/projects/Music-Dance-Video-Synthesis_pointT_firstframe/Demo/full/keypoints",
        metavar="FILE",
                                        help="path to output",
                                                type=str
                                                    )


args = parser.parse_args()

file_path=args.model
counter=args.count

output_dir=args.output
try:
    os.makedirs(output_dir)
except OSError:
    pass

keypoint_dir=args.keypoint
try:
        os.makedirs(keypoint_dir)
except OSError:
        pass

audio_path=output_dir + "/audio"
try:
    os.makedirs(audio_path)
except OSError:
    pass

Tensor = torch.cuda.FloatTensor
generator = Generator(1)
generator.eval()
generator.load_state_dict(torch.load(file_path))
generator.cuda()
data=DanceDataset(args.data)
dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=False)
criterion_pixelwise = torch.nn.L1Loss()
count = 0
total_loss=0.0
img_orig = np.ones((360,640,3), np.uint8) * 255
centers = []
scales = []
for i, (x,target) in enumerate(dataloader):
            audio_out=x.view(-1) #80000
            scaled=np.int16(audio_out)
            
            audio = Variable(x.type(Tensor).transpose(1,0))#50,1,1600
            pose = Variable(target.type(Tensor))#1,50,18,2
            pose=pose.view(1,50,36)
            
            #print('pose', pose[:,0:1].size()) [1, 1, 36]
            data = np.load('/data1/htang/alex/fomm-private/data/jian-data/test_keypoints/36#000000#000087.mp4.npy')[:,:,:2]
            data = data[0,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18], :]
            #data = 2 * data - 1
            
            #data = ((data - np.array([0.47753665, 0.46566933]).reshape((1, 1, 2))) / 0.43584909) * 0.15066236 + np.array([0.5057573, 0.42266726]).reshape((1, 1, 2))
            data = ((data - np.array([0.47753665, 0.46566933]).reshape((1, 2))) / 0.43584909) * 0.15066236 + np.array([0.5057573, 0.42266726]).reshape((1, 2))
            #data = ((data - np.array([0.47753665, 0.46566933]).reshape((1, 2))) / 0.43584909) * 0.15066236 + np.array([0.5057573, 0.42266726]).reshape((1, 2))
            data = 2 * data - 1
            data = torch.tensor(data).float().cuda()
            #data.unsqueeze(0)
            #print('pose',data.size())
            data = data.view(1,36)
            data = torch.unsqueeze(data, 0)
            #print('pose',data.size())
            # GAN loss
            #fake = generator(audio, pose[:,0:1])
            fake = generator(audio, data)
            loss_pixel = criterion_pixelwise(fake, pose)

            #start = pose[:, [8, 11]].mean(axis=1, keepdims=True)
            #end = pose[:, 1:2]
            #center = (end + start) / 2
            #scale = abs(end - start)[..., 1:]

            total_loss+=loss_pixel.item()
            
            fake = fake.contiguous().cpu().detach().numpy()#1,50,36 
            fake = fake.reshape([50,36])
            
            if(count <= counter):
                write(output_dir+"/audio/{}.wav".format(i),16000,scaled)
                real_coors = pose.cpu().numpy()
                fake_coors = fake
                real_coors = real_coors.reshape([-1,18,2])
                
                #real_coors2 = (real_coors + 1) * 0.5
                #start = real_coors2[:, [8, 11]].mean(axis=1, keepdims=True)
                #end = real_coors2[:, 1:2]
                #center = (end + start) / 2
                #scale = abs(end - start)[..., 1:]
                
                #centers.append(center)
                #scales.append(scale)

                fake_coors = fake_coors.reshape([-1,18,2])
                np.savez(keypoint_dir+"/"+str(i)+".npz", keypoint=fake_coors)

                real_coors[:,:,0] = (real_coors[:,:,0]+1) * 320
                real_coors[:,:,1] = (real_coors[:,:,1]+1 ) * 180
                real_coors = real_coors.astype(int)
                
                fake_coors[:,:,0] = (fake_coors[:,:,0]+1) * 320
                fake_coors[:,:,1] = (fake_coors[:,:,1]+1 ) * 180
                fake_coors = fake_coors.astype(int)
                
                save_2_batch_images(real_coors,fake_coors,batch_num=count,save_dir_start=output_dir)
            count += 1

#centers = np.concatenate(centers, axis=0).reshape((-1, 2))
#scales = np.concatenate(scales, axis=0).reshape((-1, 1))
#print('centers', np.mean(centers, axis=0))
#print('scales', np.mean(scales, axis=0))

final_loss=total_loss/count
print("final_loss:",final_loss)
