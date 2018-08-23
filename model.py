import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
class VAE(nn.Module):

  ### If the architecture doesn't have good performance, please modify it
  def __init__(self, latent_variable_size):
    super(VAE, self).__init__()
    # VAE Encoder
    nc = 1
    ndf = 64
    ngf = 64    
    
    self.Encoder = nn.Sequential(
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2),

      nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf*2),
      nn.LeakyReLU(0.2),

      nn.Conv2d(ndf*2, ndf*4, 7, 1, 0, bias=False),
      nn.BatchNorm2d(ndf*4),
    )
    
    self.mu = nn.Linear(ndf*4*1*1, latent_variable_size)
    self.logvar = nn.Linear(ndf*4*1*1, latent_variable_size)
    
    # VAE Decoder
    self.Decoder_input = nn.Sequential(
      nn.Linear(latent_variable_size, ndf*4*7*7, bias=False),
      nn.LeakyReLU(0.2),
    )

    self.Decoder = nn.Sequential(
      nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf*2),
      nn.LeakyReLU(0.2),

      nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
      nn.BatchNorm2d(nc),
      
      nn.Sigmoid()
    )
    
  def reparametrize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)
    
  def encode(self, x):
    x = self.Encoder(x)
    x = x.view(x.size(0), -1)
    mu, logvar = self.mu(x), self.logvar(x)
    z = self.reparametrize(mu, logvar)
    return mu, logvar, z

  def decode(self, z):
    recon_x = self.Decoder_input(z)
    recon_x = recon_x.view(recon_x.size(0), -1, 7, 7)
    recon_x = self.Decoder(recon_x)
    return recon_x

  ### need to modify in CVAE version 
  def inference(self, x):
    return self.decode(x) 

  ### need to modify in CVAE version 
  def forward(self, x):
    mu, logvar, z = self.encode(x)
    recon_x = self.decode(z)
    return recon_x, mu, logvar, z