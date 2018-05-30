import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.conv1 = nn.Conv2d(1,16,5) # H_out = 24/2 = 12
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,5) # H_out = 8/2 = 4
        self.bn2 = nn.BatchNorm2d(32)
        self.fc11 = nn.Linear(32*4*4, 10)
        self.fc12 = nn.Linear(32*4*4, 10)
        self.fc2 = nn.Linear(10,32*4*4)
        self.decode_bn1 = nn.BatchNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(32,16,5)
        self.decode_bn2 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16,1,5)
        
    def encode(self, x):
        
        x = self.bn1(F.relu(F.avg_pool2d(self.conv1(x),2)))
        x = self.bn2(F.relu(F.avg_pool2d(self.conv2(x),2)))
        x = x.view(-1, 32*4*4)
        return self.fc11(x), self.fc12(x)
    
    def decode(self, z):
        
        z = self.fc2(z)
        z = z.view(-1, 32, 4, 4)
        z = F.relu(self.deconv1(self.decode_bn1(F.upsample(z, scale_factor=2, mode='bilinear'))))
        z = F.sigmoid(self.deconv2(self.decode_bn2(F.upsample(z, scale_factor=2, mode='bilinear'))))
        return z
    
    def reparameterize(self, mu, logvar):
        
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def vae_loss(pred, x, mu, logvar):
    	BCE = F.binary_cross_entropy(pred, x, size_average=False)
    	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    	return BCE + KLD