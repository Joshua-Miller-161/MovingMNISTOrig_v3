import torch
from lightning.pytorch.callbacks import RichProgressBar
#from torchview import draw_graph
import os
#====================================================================
def GetDevice():
    if torch.cuda.is_available():
        print(" >> Found CUDA:", torch.cuda.is_available())
        return 'cuda'
    elif torch.backends.mps.is_available():
        print(" >> Found MPS:", torch.backends.mps.is_available())
        return 'mps'
    else:
        print(" >> Using CPU")
        return 'cpu'
# #====================================================================
# def plot_model(model, input_size, input_data=None, expand_nested=True, to_file=True, save_path='', model_name='model', device='cpu'):
#     if (save_path == ''):
#         save_path = os.getcwd()
    
#     model_graph = draw_graph(model, 
#                              input_size=input_size,
#                              input_data=None,
#                              device=device,
#                              show_shapes=True,
#                              expand_nested=expand_nested,
#                              roll=True,
#                              graph_name=model_name,
#                              save_graph=to_file,
#                              directory=save_path)
#     model_graph.visual_graph

#     os.remove(os.path.join(save_path, model_name+'.gv'))
#====================================================================
class CleanRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer, pl_module):
        # Get the default metrics
        items = super().get_metrics(trainer, pl_module)
        # Remove the version number
        items.pop("v_num", None)
        return items