import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from loss_sup import SupConLoss
import random

#Setting the random seed value
state = 42
def set_seeds(seed_value, use_cuda):
  np.random.seed(seed_value)  # cpu vars
  torch.manual_seed(seed_value)  # cpu  vars
  random.seed(seed_value)  # Python
  os.environ['PYTHONHASHSEED'] = str(seed_value) 
  if use_cuda:
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)  # gpu vars
      torch.backends.cudnn.deterministic = True  # needed
      torch.backends.cudnn.benchmark = False
set_seeds(state, True)

print(torch.version.cuda) #10.1
t3 = time.time()
##############################################################################
"""arguments for AE"""
args = {}
args['dim_h'] = 64          # factor controlling size of hidden layers
args['n_channel'] = 3       # number of channels in the input data 
args['n_z'] = 600           # number of dimensions in latent space.
args['sigma'] = 1.0         # variance in n_z
args['lambda'] = 0.01       # hyper param for weight of discriminator loss
args['lr'] = 0.0002         # learning rate for Adam optimizer .000
args['epochs'] = 200        # how many epochs to run for
args['batch_size'] = 100    # batch size for SGD
args['save'] = True         # save weights at each epoch of training if True
args['train'] = True        # train networks if True, else load networks from
args['temperature'] = 0.01  # temperature hyperparameter for supervised contrastive loss 
args['batch_size'] = 100    # batch size while training


def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=42):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: imbalanced_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    selected_list = []
    indices = list(range(0,length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1
    return selected_list

# create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8), 
            nn.LeakyReLU(0.2, inplace=True) )#,
        self.fc = nn.Linear(512*4, self.n_z)
    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = torch.reshape(x,(x.shape[0],2048))
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 9 * 9),
            nn.ReLU())
        
        # deconvolutional filters, essentially inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 3, 4, stride=2),
            nn.Tanh())
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 9, 9)
        x = self.deconv(x)
        return x
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(600, 350)
        self.fc2 = nn.Linear(350, 200)
        self.fc3 = nn.Linear(200, 70)
        self.fc4 = nn.Linear(70, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 100

dec_x = torch.load("data/cifar10/images.pt")
dec_y = torch.load("data/cifar10/labels.pt")

train_data = TensorDataset(dec_x,dec_y)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

encoder = Encoder(args)
decoder = Decoder(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net()
net = net.to(device)
decoder = decoder.to(device)
encoder = encoder.to(device)
temperature = args['temperature']
batch_size = args['batch_size']
train_on_gpu = torch.cuda.is_available()
criterion = nn.MSELoss()
criterion_cc = nn.CrossEntropyLoss()
criterion = criterion.to(device)
criterion_cc = criterion_cc.to(device)
criterion_supcon = SupConLoss(temperature=temperature).to(device)
num_workers = 0
best_loss = np.inf
t0 = time.time()
if args['train']:
    enc_optim = torch.optim.Adam(encoder.parameters(), lr = args['lr'])
    dec_optim = torch.optim.Adam(decoder.parameters(), lr = args['lr'])
    classifier_optim = torch.optim.Adam(net.parameters(), lr = args['lr'])

    for epoch in range(args['epochs']):
        train_loss = 0.0
        tmse_loss = 0.0
        tdiscr_loss = 0.0
        classifier_loss = 0.0
        supcon_loss = 0.0

        #Network set to training mode
        encoder.train()
        decoder.train()
        net.train()
        for images,labs in train_loader:

            # zero gradients for each batch
            encoder.zero_grad()
            decoder.zero_grad()
            net.zero_grad()
            images, labs = images.to(device), labs.to(device)
            labsn = labs.detach().cpu().numpy()

            # run images
            z_hat = encoder(images)
            train_outputs = net(z_hat)
            train_labels = labs.type(torch.int64)

            #obtaining the classifier guidance loss
            loss = criterion_cc(train_outputs, train_labels)
            z_new =  F.normalize(z_hat, dim=1)
            z_new = z_new.unsqueeze(dim=1)

            #Supervised Contrastive loss
            loss_supcon = criterion_supcon(z_new, train_labels)
            x_hat = decoder(z_hat) 

            #Mean Square error loss
            mse = criterion(x_hat,images)
            resx = []
            resy = []
            tc = np.random.choice(10,1)
            tc = tc[0]
            xbeg = dec_x[dec_y == tc]
            ybeg = dec_y[dec_y == tc] 

            xlen = len(xbeg)
            nsamp = min(xlen, 100)
            ind = np.random.choice(list(range(len(xbeg))),nsamp,replace=False)
            xclass = xbeg[ind]
            yclass = ybeg[ind]
        
            xclen = len(xclass)

            xcminus = np.arange(1,xclen)

            xcplus = np.append(xcminus,0)
            xcnew = (xclass[[xcplus],:])
            xcnew = xcnew.reshape(xcnew.shape[1],xcnew.shape[2],xcnew.shape[3],xcnew.shape[4])
        
            xcnew = torch.Tensor(xcnew)
            xcnew = xcnew.to(device)
        
            #encode xclass to feature space
            xclass = torch.Tensor(xclass)
            xclass = xclass.to(device)
            xclass = encoder(xclass)
            xclass = xclass.detach().cpu().numpy()
        
            xc_enc = (xclass[[xcplus],:])
            xc_enc = np.squeeze(xc_enc)
            xc_enc = torch.Tensor(xc_enc)
            xc_enc = xc_enc.to(device)
            
            ximg = decoder(xc_enc)
            mse2 = criterion(ximg,xcnew)
            comb = mse + mse2 + loss + loss_supcon 
            comb.backward()
        
            enc_optim.step()
            dec_optim.step()
            classifier_optim.step()
        
            train_loss += comb.item()*images.size(0)
            tmse_loss += mse.item()*images.size(0)
            tdiscr_loss += mse2.item()*images.size(0)
            classifier_loss += loss.item()*images.size(0)
            supcon_loss += loss_supcon.item()*images.size(0)

        train_loss = train_loss/len(train_loader)
        tmse_loss = tmse_loss/len(train_loader)
        tdiscr_loss = tdiscr_loss/len(train_loader)
        classifier_loss = classifier_loss/len(train_loader)
        supcon_loss = supcon_loss/len(train_loader)

        print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f}  \tclassifier loss: {:.6f} \tsuploss loss: {:.6f}'.format(epoch,
                train_loss,tmse_loss,loss, loss_supcon))

        #best model checkpoint saved
        if train_loss < best_loss:
            print('Saving..')
            path_cifar = './models/cifar10'
            if os.path.exists(path_cifar) == False:
                os.mkdir(path_cifar)
            path_encoder = path_cifar \
                 + '/encoder'+str(temperature)+'.pth'
            path_decoder = path_cifar \
                 + '/decoder'+str(temperature)+'.pth'
            torch.save(encoder.state_dict(), path_encoder)
            torch.save(decoder.state_dict(), path_decoder)
            best_loss = train_loss
    
t1 = time.time()
print('total time(min): {:.2f}'.format((t1 - t0)/60))             