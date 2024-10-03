import sys
sys.dont_write_bytecode = True
import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

sys.path.append(os.getcwd())
from misc import GetDevice, CleanRichProgressBar
from L_model import L_Model
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

model = L_Model(in_channels=1, 
                filters=9,
                kernel_size=(8, 8),
                frame_size=(x_init.shape[3], x_init.shape[4]),
                padding='same',
                activation="tanh",
                recurrent_activation='sigmoid').to(device)
#====================================================================
''' Train '''

logger = TensorBoardLogger("tb_logs", name="MovingMNIST")

trainer = L.Trainer(accelerator='gpu', 
                    devices=[0], 
                    min_epochs=1, 
                    max_epochs=num_epochs,
                    callbacks=[EarlyStopping(monitor='val_loss', 
                                             patience=5),
                               CleanRichProgressBar(leave=False),
                               LearningRateMonitor(logging_interval='epoch')],
                    logger=logger)

start = time.time()
trainer.fit(model, train_loader, val_loader)
end = time.time()
print("____________________________________________________________")
print(" >> Time:", round(end-start, 5))
print("____________________________________________________________")
#====================================================================
''' Plot '''

log_dir = logger.log_dir
event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()

all_keys = event_acc.Tags()['scalars']

# Print all keys
print("All keys (tags) in the TensorBoard log:")
for key in all_keys:
    print(key)

train_loss_values = [x.value for x in event_acc.Scalars('train_loss')]
val_loss_values   = [x.value for x in event_acc.Scalars('val_loss')]
lr_values         = [x.value for x in event_acc.Scalars('learning_rate')]
epochs            = np.arange(len(lr_values))

# Plot the losses
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(epochs, train_loss_values, label='Train Loss')
ax[0].plot(epochs, val_loss_values, label='Validation Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training and Validation Loss')
ax[0].legend()

ax[1].plot(epochs, lr_values, label='Learning rate')
ax[1].legend()
#====================================================================
print("____________________________________________________________")
print("____________________________________________________________")
print("____________________________________________________________")

#====================================================================
plt.show()