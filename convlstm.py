import sys
sys.dont_write_bytecode = True
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
from misc import GetDevice
#====================================================================
device = GetDevice()
torch.set_default_device(device)
#====================================================================
def get_activation(name):
    assert name in ['tanh', 'sigmoid', 'relu', 'leaky_relu', 'gelu'], "Activations must be:\n >> 'tanh', 'sigmoid', 'relu', 'leaky_relu', 'gelu' <<\n Got: "+str(name)
    if (name == 'tanh'):
        return F.tanh
    elif (name == 'sigmoid'):
        return F.sigmoid
    elif (name == 'relu'):
        return F.relu
    elif (name == 'leaky_relu'):
        return F.leaky_relu
    elif (name == 'gelu'):
        return F.gelu
#====================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self,
                 in_channels,
                 filters, 
                 kernel_size,
                 frame_size,
                 padding='same',
                 activation='tanh',
                 recurrent_activation='sigmoid'):
        super(ConvLSTMCell, self).__init__()
        
        self.activation           = get_activation(activation)
        self.recurrent_activation = get_activation(recurrent_activation)

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(in_channels=in_channels + filters, 
                              out_channels=4 * filters, 
                              kernel_size=kernel_size, 
                              padding=padding)           

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(filters, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(filters, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(filters, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate  = self.recurrent_activation(i_conv + self.W_ci * C_prev)
        forget_gate = self.recurrent_activation(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = self.recurrent_activation(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C
#====================================================================
class ConvLSTM_(nn.Module):
    def __init__(self,
                 in_channels,
                 filters, 
                 kernel_size,
                 frame_size,
                 padding='same',
                 activation='tanh',
                 recurrent_activation='sigmoid'):
        super(ConvLSTM_, self).__init__()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Args
        self.filters = filters
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Layers
        self.convLSTMcell = ConvLSTMCell(in_channels,
                                         filters, 
                                         kernel_size,
                                         frame_size,
                                         padding=padding,
                                         activation=activation,
                                         recurrent_activation=recurrent_activation)
        
    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.filters, seq_len, height, width, device=device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.filters, height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.filters, height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output, C
#====================================================================
class ConvLSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 frame_size,
                 padding='same',
                 data_format='channels_last',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 return_sequences=False,
                 return_state=False):
        super(ConvLSTM, self).__init__()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Args
        self.in_channels      = in_channels
        self.filters          = filters
        self.kernel_size      = kernel_size
        self.padding          = padding
        self.data_format      = data_format
        self.return_sequences = return_sequences
        self.return_state     = return_state

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Layer
        self.convlstm = ConvLSTM_(in_channels=in_channels,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding, 
                                  frame_size=frame_size,
                                  activation=activation,
                                  recurrent_activation=recurrent_activation)

    def forward(self, X):
        output, current_state = self.convlstm(X)
        
        if (self.return_sequences and self.return_state):
            return output, current_state
        
        elif self.return_sequences:
            return output
        
        elif self.return_state:
            return current_state
        
        else:
            return output[:, :, -1] # Return only the last output frame