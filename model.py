import sys
sys.dont_write_bytecode = True
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
from misc import GetDevice
from convlstm import ConvLSTM
#====================================================================
device = GetDevice()
torch.set_default_device(device)
#====================================================================
class Model(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 frame_size,
                 padding='same',
                 data_format='channels_last',
                 activation='tanh',
                 recurrent_activation='sigmoid'):
        super(Model, self).__init__()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Args
        self.in_channels      = in_channels
        self.filters          = filters
        self.kernel_size      = kernel_size
        self.padding          = padding
        self.data_format      = data_format

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Layers
        self.convlstm1 = ConvLSTM(in_channels=in_channels,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  frame_size=frame_size,
                                  activation=activation,
                                  recurrent_activation=recurrent_activation,
                                  return_sequences=True)
        
        self.norm1     = nn.BatchNorm3d(num_features=filters)

        self.convlstm2 = ConvLSTM(in_channels=filters,
                                  filters=filters*2,
                                  kernel_size=kernel_size,
                                  padding=padding, 
                                  frame_size=frame_size,
                                  activation=activation,
                                  recurrent_activation=recurrent_activation,
                                  return_sequences=True)
        
        self.norm2     = nn.BatchNorm3d(num_features=filters*2)

        self.convlstm3 = ConvLSTM(in_channels=filters*2,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding, 
                                  frame_size=frame_size,
                                  activation=activation,
                                  recurrent_activation=recurrent_activation,
                                  return_state=True)
        
        self.norm3     = nn.BatchNorm3d(num_features=filters)

        self.conv_out  = nn.Conv3d(in_channels=filters,
                                   out_channels=1,
                                   padding=padding,
                                   kernel_size=(1, *kernel_size))
    #----------------------------------------------------------------
    def forward(self, x):
        x = self.convlstm1(x)
        x = self.norm1(x)
        x = self.convlstm2(x)
        x = self.norm2(x)
        x = self.convlstm3(x)
        x = torch.unsqueeze(x, dim=2)
        x = self.norm3(x)
        x = F.sigmoid(self.conv_out(x))

        return x