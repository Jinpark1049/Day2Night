import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
from utils.util import *
from utils.ddp import *
import wandb

# os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3" 

class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # gpu setting/ ddp/ seed initialization
        self.parser.add_argument('--distributed', action='store_true', help='if use ddp, default = False') # torchrun --nnodes=1 --nproc-per-node=4 main_wb.py --distributed
        self.parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--seed', type=int, default=42, help='set seed')
        
        # wandb
        self.parser.add_argument('--wandb', action='store_true', help='if use wandb, default = False') # wandb login -> add flag --wandb
        self.parser.add_argument('--proj_name', type=str, default='d2n_gan', help='specify the name of the wandb project')

        # experiments
        self.parser.add_argument('--name', type=str, default ='cycleGAN_256x256_vfp290k',help='name of the experiment. It decides where to store samples and models') # cycleGAN_256x256_vfp290k_tl
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model_name', type=str, default ='cycleGAN',help='name of the model ex) cycleGAN, Adain')

        # dataset
        self.parser.add_argument('--vfp290k', action='store_false', help='if use vfp290k dataset, default = False') # wandb login -> add flag --wandb

        """
        dataset directory
        data
            bdd100k
                    train
                            A
                            B
                    test
                            A 
                            B
        """ 
        self.parser.add_argument('--num_workers', default=32, type=int, help='Number of workers used in dataloading') #  cat /proc/cpuinfo | grep -c processor
        self.parser.add_argument('--batchsize', type=int, default=16, help='batch_size for training')
        self.parser.add_argument('--dataroot',  type=str, default='/root/jin/model/data/bdd100k',help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size when loading')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize|resize_and_crop|crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--cropsize', type=int, default=256, help='then crop to this size')
        
                
        
        self.initialized = True    

    def print_options(self):

        print('------------ Options -------------')
        for k, v in sorted(self.args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def save_options(self):
        # save args to the disk

        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(self.args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.args = vars(self.opt)
        
        self.opt.isTrain = self.isTrain   # train or test
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids or ddp
        if len(self.opt.gpu_ids) > 0:
            if self.opt.distributed:
                init_distributed_mode(self.opt)
                torch.cuda.set_device(get_rank())
                self.opt.seed = self.opt.seed + get_rank() # rank 별 seed
                if get_rank() == 0:
                    self.print_options()
                    self.save_options()
                    if self.opt.wandb:
                        wandb.init(project=self.opt.proj_name, name=self.opt.name, entity="parkjy2", config=self.args)   # change entity to your user name 
            else:
                torch.cuda.set_device(self.opt.gpu_ids[0]) # let main to 0.
                self.print_options()
                self.save_options()
                if self.opt.wandb:
                    wandb.init(project=self.opt.proj_name, name=self.opt.name, entity="parkjy2", config=self.args)
        # set seed
        seed_everything(self.opt.seed)
        
        return self.opt


