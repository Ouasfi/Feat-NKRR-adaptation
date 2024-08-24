import poco.model as poco
import torch
import yaml
import re
from adapters import datamodule
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        #cfg_special = yaml.full_load(f)
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')
    #print(inherit_from)
    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg
def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


   
   


def load_poco(conf, default_conf = '/home/amine/POCO/configs/config_default.yaml'):
    
    config = load_config(conf, default_conf)
    config = eval(str(config))    
    device = torch.device(config['device'])
    model = poco.POCONetwork(in_channels= 3, latent_size=config["network_latent_size"], out_channels=2,
                                    backbone = 'FKA_Conv_Encoder',
                                    decoder = {"name": "Attention_KHeads_Decoder", "k": config['network_decoder_k']}).cuda()
  

    model.to(device);
    checkpoint = 'checkpoint.pth'
    checkpoint = torch.load(conf.replace('config.yaml',checkpoint ), map_location=device)["state_dict"]
    state_dict = {re.sub(r'net\.', 'encoder.', k, count=1).replace('projection', 'decoder'):v for k,v in checkpoint.items()}
    model.load_state_dict(state_dict)
    model.eval();
    model.encoder.eval()
    model.decoder.eval()
    model.feat_ctx  = datamodule.POCO_Feature_Context

    model.feat_fn = datamodule.compute_features 
 


    return model