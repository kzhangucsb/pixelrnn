
"""
Created on Sun Apr 28 15:35:31 2019

@author: zkq
"""

import torch 
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import os.path

#import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
num_epochs = 50
prelinear_dim = 32
hidden_dim = 128
batch_size = 100
learning_rate = 0.001
modelfname = 'pixel_rnn.ckpt'
plot_img = True

train_data = np.ndarray([0, 3, 32, 32], dtype=np.uint8)
train_label = np.ndarray(0, dtype=int)
for batch in range(1, 6):
    with open('cifar-10-batches-py/data_batch_{}'.format(batch), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    img = np.reshape(d[b'data'], (-1, 3, 32, 32))
    labels = d[b'labels']
    train_data  = np.concatenate((train_data, img))
    train_label = np.concatenate((train_label, labels))
    
with open('cifar-10-batches-py/test_batch', 'rb') as f:
    d = pickle.load(f, encoding='bytes')
test_data = np.reshape(d[b'data'], (-1, 3, 32, 32))
test_label = d[b'labels']
    
    
    
train_data_normalized = torch.FloatTensor(train_data/256.0)
train_label           = torch.LongTensor(train_data)
test_data_normalized  = torch.FloatTensor(test_data/256.0)
test_label            = torch.LongTensor(test_data)

train_dataset = TensorDataset(train_data_normalized, train_label)
test_dataset  = TensorDataset(test_data_normalized, test_label)


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural network
def mse_crossentropy_loss(value, target):
    # value: [batch, N]
    # targer: [batch]
    value = nn.functional.log_softmax(value, dim=1)
    error = torch.zeros_like(value)
    for t in range(value.shape[1]):
        error[:, t] = -nn.functional.relu(16-abs(t - target.float())) * value[:, t]
    return torch.mean(error, [0, 1])
           

class pixelGRUlayer(nn.Module):
    def __init__(self, hidden_dim, prelinear_dim, channels=3, v_max = 256, layers = 1):
        super(pixelGRUlayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.prelinear_dim = prelinear_dim
        self.channels = channels
        self.v_max = v_max
        self.layers = layers
        self.pre_linear = nn.Sequential(
            nn.Linear(channels*5-1, prelinear_dim),
            nn.Sigmoid()
        )
        self.gru = nn.GRU(prelinear_dim, hidden_dim, layers, batch_first=True) # batch, seq, feature
        self.post_linear = nn.Linear(hidden_dim, v_max)
        
        
    def _img2input(self, x):
        # x : (batch, channel, h, w)
        # y : (batch, (h-2)*(w-2)*channel, channel*5-1)
        [b, c, h, w] = x.shape
        
        x = x.permute(0, 2, 3, 1)
        y = torch.zeros([b, h-2, w-2, c, c*5-1]).to(x.device)
        
        for ic in range(c):
            # b,h,w,c, feature
            y[:,:,:,ic,  0:  c] = x[:, 0:-2, 0:-2, :] # topleft
            y[:,:,:,ic,  c:2*c] = x[:, 0:-2, 1:-1, :] # top
            y[:,:,:,ic,2*c:3*c] = x[:, 0:-2, 2:  , :] # topright
            y[:,:,:,ic,3*c:4*c] = x[:, 1:-1, 0:-2, :] # left
            for ic1 in range(c-1):
                if (ic1 < ic):
                    y[:,:,:,ic,4*c+ic1] = x[:, 1:-1, 1:-1, ic1] # this pixel
                else:
                    y[:,:,:,ic,4*c+ic1] = -1
        return y.reshape([b, -1, c*5-1])
    
    def forward(self, x):
        # x : (batch, channel, h, w)
        # y : (batch, channel, h, w, value)
        # space for status
        [b, c, h, w] = x.shape
        assert(c == self.channels)
            
         
        if self.training:
            x = self._img2input(x)
            x = self.pre_linear(x)
            (y, s) = self.gru(x)
            y = self.post_linear(y)

            y = y.reshape([b, h-2, w-2, c, self.v_max])
            y = y.permute([0, 3, 1, 2, 4])
            # padding
            y = nn.functional.pad(y, [0,0,1,1,1,1])
        else:
            x = nn.functional.pad(x, [1,1,1,1])
            s = torch.zeros(self.layers, b, self.hidden_dim).to(x.device) # initial state
            y = torch.zeros(b, c, h+2, w+2, self.v_max).to(x.device) #output
            # scanning
            def batch_fill_guess(x, y):
                z = torch.where(x < 0, torch.argmax(y, 1).float() / 256.0, x)
                return z
                
                
            for ih in range(h):
                for iw in range(w):
                    for ic in range(c):
                        # input
                        gru_input = torch.zeros([b, 1, c*5-1]).to(x.device)
                        for ic1 in range(c):
                            gru_input[:,0, ic1    ] = batch_fill_guess(
                                    x[:,ic1, ih  , iw  ], y[:,ic1, ih  , iw  , :]) # topleft
                            gru_input[:,0, ic1+  c] = batch_fill_guess(
                                    x[:,ic1, ih  , iw+1], y[:,ic1, ih  , iw+1, :]) # top
                            gru_input[:,0, ic1+2*c] = batch_fill_guess(
                                    x[:,ic1, ih  , iw+2], y[:,ic1, ih  , iw+2, :]) # topright
                            gru_input[:,0, ic1+3*c] = batch_fill_guess(
                                    x[:,ic1, ih+1, iw  ], y[:,ic1, ih+1, iw  , :]) # left
                            if (ic1 < ic):
                                gru_input[:,0,ic1+4*c] = batch_fill_guess(
                                        x[:,ic1, ih+1, iw+1], y[:,ic1, ih+1, iw+1, :]) # this
                            elif (ic1 < c - 1):
                                gru_input[:,0,ic1+4*c] = -1
                        
                        # gru
                        gru_input = self.pre_linear(gru_input)
                        (o, s) = self.gru(gru_input, s)
                        o = self.post_linear(o)
                        y[:, ic, ih+1, iw+1, :] = o.reshape(b, self.v_max)
            y = y[:, :, 1:-1, 1:-1, :] 
        return y
        


class PixelGRUNet(nn.Module):
    def __init__(self, hidden_dim, prelinear_dim):
        super(PixelGRUNet, self).__init__()
        self.grulayer = pixelGRUlayer(hidden_dim, prelinear_dim)
        
    def forward(self, x):
        out = self.grulayer(x)
        return out

model = PixelGRUNet(hidden_dim, prelinear_dim).to(device)

# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
criterion = mse_crossentropy_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
t_run = np.zeros(num_epochs)
train_loss = np.zeros(num_epochs)
test_loss = np.zeros(num_epochs)


total_step = len(train_loader)

if os.path.isfile(modelfname):
    #load model
    model.load_state_dict(torch.load(modelfname, map_location=device))
    print("model loaded!")
else:
    # training
    model.train()
    for epoch in range(num_epochs):
        t_start = time.time()
        
        loss_bat = np.zeros(total_step)
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs.reshape([-1, 256]), labels.reshape([-1]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 1 == 0:
                print ('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()), end='')
            loss_bat[i] = loss.item()
        t_run[epoch] = time.time() - t_start
        train_loss[epoch] = np.mean(loss_bat)
        
        with torch.no_grad():
            loss_bat = np.zeros(len(test_loader))
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs.reshape([-1, 256]), labels.reshape([-1]))
                loss_bat[i] = loss.item()
            test_loss[epoch] = np.mean(loss_bat)
            print ('\nEpoch [{}/{}], testing set, Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, loss.item()))
        torch.save(model.state_dict(), modelfname.split('.')[0]+'_progress/epoch{}.ckpt'.format(epoch))
    torch.save(model.state_dict(), modelfname)
    with open('train_loss.pickle', 'wb') as f:
        pickle.dump({'train_loss': train_loss, 'test_loss': test_loss, 'runtime': t_run}, f)
#
    
#        
#    
#    
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
loss_calculator = nn.MSELoss()
with torch.no_grad():
    image_recoverd_all = []
    loss_bat = np.zeros(len(test_loader))
    for i, (images, labels) in enumerate(test_loader):
        images[:,:,5:, 5::2] = -1
        images[:,:,5::2, 5:] = -1# blank out
        images = images.to(device)
        outputs = model(images)
        image_recoverd = torch.argmax(outputs, -1)
        image_recoverd_all.append(image_recoverd)

        labels = labels.to(device)
        #error = torch.mean((image_recoverd.float() - labels.float())**2, (0,1,2,3))**0.5
        error = loss_calculator(image_recoverd.float(), labels.float()).item()
        if (plot_img):
            try:
                image_recoverd = image_recoverd.permute(0, 2, 3, 1)
                image_recoverd = np.uint8(image_recoverd.numpy())
                plt.figure(figsize=[10, 10])
                plt.subplots_adjust(hspace=0, wspace=0)
                for j in range(100):
                    plt.subplot(10, 10, j+1)
                    plt.imshow(image_recoverd[j])
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.gca().axes.get_yaxis().set_visible(False)
                plt.show()
            except:
                pass
        print('\rTesting [{}/{}] MSE {}'.format(i+1, len(test_loader), error), end='')
        loss_bat[i] = error
    print('\nTesting finished!')
    image_recoverd_all = torch.cat(image_recoverd_all)
    with open('images_recovered.pickle', 'wb') as f:
        pickle.dump({'img': image_recoverd_all.cpu(), 'loss': loss_bat}, f)

