import time
from options.train_options import TrainOptions
from data.data_loader import DataLoader
from models.cyclegan_model import Cyclegan
from models.adain import Adain # 완전 바꿔야함,
from utils import ddp
from tqdm import tqdm
import os
import copy
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def run(opt):

    dataset = DataLoader(opt) 
    print("# training images {}".format(len(dataset)*opt.batchsize))
    # for loading test dataset
#   test_opt = copy.deepcopy(opt)
#   test_opt.isTrain = False
#   test_opt.phase = 'test' 
    
#   test_dataset = DataLoader(test_opt)
#   print("# testing images {}".format(len(test_dataset)*test_opt.batchsize))
    
    if opt.model_name == 'cycleGAN':
        model = Cyclegan(opt)
    elif opt.model_name == 'Adain':
        model = Adain(opt)
    print(model.name())
    
    for epoch in range(opt.start_epoch, opt.n_epochs):
        epoch_start_time = time.time()
        pbar = tqdm(dataset, desc= 'Batch Running', unit="batch", disable=not ddp.is_main_process())
        # update learning rate
        model.update_learning_rate(epoch)
        
        for i, data  in enumerate(pbar):
            model.run_batch(epoch, i, dataset, data)

        # save model for each epoch
        print(('Total elapsed time for epoch:' , time.time() - epoch_start_time))
        if model.device == 0:
            model.model_save(epoch+1)
        # vis epoch losses
        model.get_current_losses(epoch, i, len(dataset), epoch_vis=True, test_dataset=None) # includes testing results if test dataset is not None.
        model.initialize_training()

if __name__ == '__main__': # adain update
    """
    """
    # for ddp
    # torchrun --nnodes=1 --nproc-per-node=4 main.py --distributed
    # for ddp & wandb
    #torchrun --nnodes=1 --nproc-per-node=4 main.py --distributed --wandb
    opt = TrainOptions().parse()
    
    run(opt) 