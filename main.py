import os
import argparse
from trainer import Solver
from torch.backends import cudnn
from torchvision import transforms, datasets
import torch.utils.data as data
import torch
from torchvision.utils import save_image
from datasets_torch import TextDataset


def main(config):
    cudnn.benchmark = True
    torch.manual_seed(7) # cpu
    torch.cuda.manual_seed_all(999) #gpu

    # Create directories if not exist.
    config.log_dir = os.path.join(config.model_dir, 'logs')
    config.model_save_dir = os.path.join(config.model_dir, 'models')
    config.sample_dir = os.path.join(config.model_dir, 'samples')
    config.result_dir = os.path.join(config.model_dir, 'results')
    
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # dataloader
    dataset = TextDataset(config.datadir, 'cnn-rnn', config.image_size)
    filename_test = '%s/test' % (config.datadir)
    dataset.test = dataset.get_data(filename_test)
    filename_train = '%s/train' % (config.datadir)
    dataset.train = dataset.get_data(filename_train)

    # Solver for training and testing ZstGAN.
    solver = Solver(dataset, config)

    if config.mode == 'train':
        solver.train() # train mode for ZstGAN
    elif config.mode == 'test':
        solver.test() # test mode for ZstGAN



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=200, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--n_blocks', type=int, default=0, help='number of res conv layers in C')
    parser.add_argument('--lambda_mut', type=float, default=10, help='weight for multual information  loss')
    parser.add_argument('--lambda_rec', type=float, default=1, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--ft_num', type=int, default=2048, help='number of ds feature')
    parser.add_argument('--nz_num', type=int, default=312, help='number of noise feature')
    parser.add_argument('--att_num', type=int, default=1024, help='number of attribute feature')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=300000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--ev_ea_c_iters', type=int, default=80000, help='number of iterations for training encoder_a and encoder_v')
    parser.add_argument('--c_pre_iters', type=int, default=20000, help='number of  iterations for pre-training C')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=300000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Directories.
    parser.add_argument('--datadir', type=str, default='Data/birds')
    parser.add_argument('--model_dir', type=str, default='zstgan')

    # Step size.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=20000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
