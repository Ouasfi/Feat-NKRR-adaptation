#import convonet.unets as unets
#from unets import UNet, UNet3D
import convonet.model as cvnet
from torch.utils import model_zoo
from adapters import datamodule
def load_convonet(conf, default_conf = '/home/amine/convolutional_occupancy_networks/configs/default.yaml'):
    cfg =cvnet. load_config(conf, default_conf)
    cfg['pretrained']= {'model_file': cfg['test']['model_file'] }

    encoder = cvnet.LocalPointnet_Encoder(dim = cfg['data']['dim'], 
                                        c_dim = cfg['model']['c_dim'], 
                                        padding=cfg['data']['padding'],
                                        **cfg['model']['encoder_kwargs'])
    decoder = cvnet.Local_Decoder(dim = cfg['data']['dim'], 
                                    c_dim = cfg['model']['c_dim'], 
                                    padding=cfg['data']['padding'],
                                    **cfg['model']['decoder_kwargs'])
    model = cvnet.ConvolutionalOccupancyNetwork ( decoder = decoder, encoder = encoder , device = "cuda")
    #model.eval()
    state_dict = model_zoo.load_url(cfg['pretrained']['model_file'], progress=False)['model']
    model.load_state_dict(state_dict)

    model.feat_ctx  = datamodule.CONVONET_Feature_Context
    model.feat_fn = datamodule.compute_features_convonet
    return model