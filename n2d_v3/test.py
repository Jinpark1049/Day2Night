import time
from options.test_options import TestOptions
from data.data_loader import DataLoader
from models.cyclegan_model import *
from models.adain import *
from utils import ddp
from tqdm import tqdm
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def run(opt):

    dataset = DataLoader(opt) 
    print("# test images {}".format(len(dataset)*opt.batchsize))
    # for loading test dataset
    if opt.model_name == 'cycleGAN':
        model = Cyclegan(opt)
    elif opt.model_name == 'Adain':
        model = Adain(opt)
    print(model.name())
    
    pbar = tqdm(dataset, desc= 'Test Running', unit="batch", disable=not ddp.is_main_process())
    process_time = time.time()

    for i, data in enumerate(pbar):
        model.run_test(i, data, save=True)  # 저장 기능
        if i % 30 == 0  and i> 1:
           break
    print(('Total elapsed time for processing:' , time.time() - process_time))


if __name__ == '__main__':
    opt = TestOptions().parse()
    """
    please specify which model to use from base options/base_options.py
    run python test.py
    """
    opt.start_epoch = 200 # 200, 37
    opt.batchsize = 1
    opt.loadSize = 512
    opt.cropsize = 512
    opt.name = 'cycleGan_StyleLoss_256x256' #  cycleGan_StyleLoss_256x256, Adain_256x256
    
    run(opt)