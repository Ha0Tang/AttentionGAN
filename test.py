#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--save_name', type=str, default='ar_neutral2happiness')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
# parser.add_argument('--generator_A2B', type=str, default='%s/%s' % (opt.save_name, 'netG_A2B.pth'), help='A2B generator checkpoint file')
# parser.add_argument('--generator_B2A', type=str, default='%s/%s' % (opt.save_name, 'netG_B2A.pth'), help='B2A generator checkpoint file')



opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator()
netG_B2A = Generator()

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts


netG_A2B.load_state_dict(torch.load('%s/%s' % (opt.save_name, 'netG_A2B.pth')))
netG_B2A.load_state_dict(torch.load('%s/%s' % (opt.save_name, 'netG_B2A.pth')))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('%s/%s' % (opt.save_name, 'testing')):
    os.makedirs('%s/%s' % (opt.save_name, 'testing'))

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B, mask_B, temp_B = netG_A2B(real_A)
    fake_B_1 = 0.5*fake_B.data[0] + 0.5
    fake_B_2 = 0.5*temp_B.data[0] + 0.5
    fake_A, mask_A, temp_A = netG_B2A(real_B)
    fake_A_1 = 0.5*fake_A.data[0] + 0.5
    fake_A_2 = 0.5*temp_A.data[0] + 0.5
    # Save image files
# '%s/%s' % (opt.save_name, 'testing')

    save_image(real_A.data.cpu()[0]*0.5+0.5, '%s/%s/%04d_real_A.png' % (opt.save_name, 'testing', i+1))
    save_image(real_B.data.cpu()[0]*0.5+0.5, '%s/%s/%04d_real_B.png' % (opt.save_name, 'testing',i+1))
    save_image(fake_A_1, '%s/%s/%04d_fake_A_1.png' % (opt.save_name, 'testing',i+1))
    save_image(fake_B_1, '%s/%s/%04d_fake_B_1.png' % (opt.save_name, 'testing',i+1))
    save_image(fake_A_2, '%s/%s/%04d_fake_A_2.png' % (opt.save_name, 'testing',i+1))
    save_image(fake_B_2, '%s/%s/%04d_fake_B_2.png' % (opt.save_name, 'testing',i+1))
    save_image(mask_A.data.cpu()[0], '%s/%s/%04d_mask_A.png' % (opt.save_name, 'testing',i+1))
    save_image(mask_B.data.cpu()[0], '%s/%s/%04d_mask_B.png' % (opt.save_name, 'testing',i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
