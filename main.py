import sys
sys.dont_write_bytecode = True
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from misc import *
from convlstm import ConvLSTM
from model import Model
#====================================================================
device = GetDevice()
torch.set_default_device(device)
#====================================================================
''' Hyperparams '''
num_epochs = 2
in_frames  = 10
out_frames = 1
filters    = 1
#====================================================================
''' Data '''
parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--path', type=str, nargs=1,
                    default='/Users/joshuamiller/Python Files/MovingMNISTPyTorch/data/MovingMNIST/mnist_test_seq.npy', required=False)
parser.add_argument('--num_epochs', type=int, nargs='?',
                    default=2, required=False)

args = parser.parse_args()
path       = args.path
num_epochs = args.num_epochs


MovingMNIST = np.load(path).transpose(1, 0, 2, 3)
print(" >> MovingMNIST:", MovingMNIST.shape)
print(" >> num_epochs", num_epochs)
# Shuffle Data
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:8000]         
val_data = MovingMNIST[8000:9000]       
test_data = MovingMNIST[9000:10000]     

def collate(batch):

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)     
    batch = batch / 255.0                        
    batch = batch.to(device)                     

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(in_frames,20)                     
    return batch[:,:,rand-in_frames:rand], batch[:,:,rand]     

# Training Data Loader
train_loader = DataLoader(train_data, shuffle=True, 
                          batch_size=16, collate_fn=collate,
                          generator=torch.Generator(device=device))

# Validation Data Loader
val_loader = DataLoader(val_data, shuffle=True, 
                        batch_size=16, collate_fn=collate, 
                        generator=torch.Generator(device=device))

x_init, y_init = next(iter(train_loader))
print(" >> x_init", x_init.shape, x_init.device, ", y_init", y_init.shape, y_init.device)
#====================================================================
''' Model '''

# model = ConvLSTM(in_channels=1, 
#                  filters=1,
#                  kernel_size=(3, 3),
#                  frame_size=(x_init.shape[3], x_init.shape[4]),
#                  padding='same',
#                  activation="tanh",
#                  recurrent_activation='sigmoid').to(device)

model = Model(in_channels=1, 
              filters=9,
              kernel_size=(8, 8),
              frame_size=(x_init.shape[3], x_init.shape[4]),
              padding='same',
              activation="tanh",
              recurrent_activation='sigmoid').to(device)

optim     = Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss(reduction='sum')

x_test = torch.rand_like(x_init)
y_test = model(x_test)
print(" >> y_test", y_test.shape, y_test.device)

print(" >> Devices: model", next(model.parameters()).device, ", x", x_init.device, ", y", y_init.device)

summary(model, input_size=x_init.shape, col_names=("input_size", "output_size", "num_params"), verbose=1, depth=7, device=device)

#plot_model(model, x_init.shape, model_name='MovingMNIST', device=device)

#====================================================================
for epoch in range(1, num_epochs+1):
    
    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        output = model(input)                                     
        loss = criterion(output.flatten(), target.flatten())       
        loss.backward()                                            
        optim.step()                                               
        optim.zero_grad()                                           
        train_loss += loss.item()                                 
    train_loss /= len(train_loader.dataset)                       

    val_loss = 0                                                 
    model.eval()                                                   
    with torch.no_grad():                                          
        for input, target in val_loader:                          
            output = model(input)                                   
            loss = criterion(output.flatten(), target.flatten())   
            val_loss += loss.item()                                
    val_loss /= len(val_loader.dataset)                            

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))


#====================================================================
''' Plot results '''
x_test, y_test = next(iter(val_loader))
y_pred = model(x_test)
print(" >> x_test", x_test.shape, x_test.device, ", y_test", y_test.shape, y_test.device, ", y_pred", y_pred.shape, y_pred.device)

x_test = x_test.detach().cpu().numpy()
y_test = y_test.detach().cpu().numpy()
y_pred = y_pred.detach().cpu().numpy()

input_frames = x_test[0]
true_frames  = y_test[0]
pred_frames  = y_pred[0]
print(" >> input_frames", input_frames.shape, ", true_frames", true_frames.shape, ", pred_frames", pred_frames.shape)

#--------------------------------------------------------------------
fig, ax = plt.subplots(3, max([in_frames, out_frames]))
plt.xticks([])
plt.yticks([])

for i in range(input_frames.shape[1]):
    ax[0][i].imshow(np.squeeze(input_frames[:, i, :, :]) * 255, cmap='gray')

for j in range(true_frames.shape[0]):
    ax[1][j].imshow(np.squeeze(true_frames[j]) * 255, cmap='gray')

for k in range(true_frames.shape[0]):
    ax[2][k].imshow(np.squeeze(pred_frames[k]) * 255, cmap='gray')

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
ax[0][0].set_ylabel('Input frames')
ax[1][0].set_ylabel('Next frame(s)')
ax[2][0].set_ylabel('Predicted frame(s)')

plt.show()
#====================================================================


#====================================================================