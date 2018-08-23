import scipy
import scipy.misc
import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from utils import save_img
from model import VAE
from torch.autograd import Variable

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD



def main(args):

    ### VAE on MNIST
    n_transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST('data', transform=n_transform, download=True)

    ### CVAE on MNIST
    # n_transform = transforms.Compose([transforms.ToTensor()])
    # dataset = MNIST('data', transform=n_transform)

    ### CVAE on facescrub-5
    # n_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    # dataset = ImageFolder('facescrub-5', transform=n_transform)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    vae = VAE(args.latent_size).cuda() 

    ### CVAE
    # vae = CVAE(args.latent_size, num_labels=args.num_labels).cuda()

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    # Decide if you want to use fixed noise or not
    fix_noise = Variable(torch.randn((50, args.latent_size)).cuda())

    num_iter = 0

    for epoch in range(args.epochs*10):
        for _, batch in enumerate(data_loader, 0):
            img = Variable(batch[0].cuda())
            label = Variable(batch[1].cuda())

            recon_img, mean, log_var, z = vae(img)

            ### CVAE
            # recon_img, mean, log_var, z = vae(img, label)
            
            loss = loss_fn(recon_img, img, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_iter += 1

            if num_iter % args.print_every == 0:
                print("Batch %04d/%i, Loss %9.4f"%(num_iter, len(data_loader)-1, loss.data.item()))

            if num_iter % args.save_test_sample == 0:
                x = vae.inference(fix_noise)
                save_img(args, x.detach(), num_iter)

            if num_iter % args.save_recon_img == 0:
                save_img(args, recon_img.detach(), num_iter, recon=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--figroot", default='fig')
    parser.add_argument("--display_row", default=5)
    parser.add_argument("--save_test_sample", type=int, default=100)
    parser.add_argument("--save_recon_img", type=int, default=100)

    parser.add_argument("--epochs", type=int, default=10)  
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--img_size", default=28)
    parser.add_argument("--dataroot", default='facescrub-5')    
    parser.add_argument("--img_channel", default=1)
    parser.add_argument("--num_labels", default=10)
    args = parser.parse_args()

    main(args)
