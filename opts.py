import argparse
def krr_opts():    
    parser = argparse.ArgumentParser(description="Reconstruction with KRR")
    parser.add_argument('--sigma', '-s', type=float, default=0.01)
    parser.add_argument('--penalty', '-p', type=float, default=0.01)
    parser.add_argument('--n_rff', '-rff', type=float, default=1024*5)
    parser.add_argument('--n_nystrom', '-n', type=int, default=500)
    parser.add_argument('--solver', type=str, default="KRR")
    parser.add_argument('--vis', '-v',  action='store_true')
    parser.add_argument('--optim_sigma', '-optim_sigma',  action='store_true')
    parser.add_argument('--save', '-save',  action='store_true')
    parser.add_argument('--silent', '-silent',  action='store_true')
    parser.add_argument('--optim_penalty', '-optim_penalty',  action='store_true')
    parser.add_argument('--optim_nystrom', '-optim_nystrom',  action='store_true')
    parser.add_argument('--normalize', '-normalize',  action='store_true')
    #parser for n_points and sigma
    parser.add_argument('--n_points', '-n_points', type=int, default=10000)
    parser.add_argument('--sigma_pc', '-sigma_pc', type=float, default=0.005)
    #parser for file id and classe
    parser.add_argument('--id', '-id', type=int, default=0)
    parser.add_argument('--classe', '-classe', type=str, default='03636649')
    #parser for backnone 
    
    parser.add_argument('--split_file', '-split_file', type=str, default='test_optim.lst')
    parser.add_argument('--backbone', '-backbone', type=str, default='convonet')
    parser.add_argument('--dataset', '-dataset', type=str, default='shapenet')
    ## add argument for root
    parser.add_argument('--root', '-root', type=str, default='/home/amine/data/ShapeNet/')
    return parser