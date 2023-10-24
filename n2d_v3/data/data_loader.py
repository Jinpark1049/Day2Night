import torch.utils.data
from data.unaligned_dataset import UnalignedDataset, VFPDataset
from utils.ddp import *


class DataLoader():
    def name(self):
        return 'DataLoader'

    def __init__(self, opt):
        self.opt = opt
        if opt.vfp290k:
            self.dataset = VFPDataset(opt)
            print((self.dataset.name()))        
        else:
            self.dataset = UnalignedDataset(opt)
            print((self.dataset.name()))     
        if self.opt.isTrain:
            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                batch_size=opt.batchsize,
                num_workers=int(opt.num_workers),
                shuffle=True,
                pin_memory=True,
                drop_last=True)
        
            if opt.distributed:
                self.sampler_train =  torch.utils.data.DistributedSampler(self.dataset, 
                                                                          num_replicas=get_world_size(), 
                                                                          rank=get_rank(), shuffle=True)

                self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                            batch_size=int(opt.batchsize / get_world_size()),
                                                            num_workers=int(opt.num_workers /get_world_size()),
                                                            sampler=self.sampler_train,
                                                            pin_memory=True,
                                                            drop_last=True)
        else:
            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                batch_size=opt.batchsize,
                num_workers=int(opt.num_workers),
                shuffle=False,
                pin_memory=True,
                drop_last=True)
        
            if opt.distributed:
                self.sampler_train =  torch.utils.data.DistributedSampler(self.dataset, 
                                                                          num_replicas=get_world_size(), 
                                                                          rank=get_rank(), shuffle=False)

                self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                            batch_size=int(opt.batchsize / get_world_size()),
                                                            num_workers=int(opt.num_workers /get_world_size()),
                                                            sampler=self.sampler_train,
                                                            pin_memory=True,
                                                            drop_last=True)
        
    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data  # yield returns the generator



      