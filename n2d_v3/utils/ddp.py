import torch
import torch.distributed as dist
import os


def is_main_process():
    return get_rank() == 0 or get_world_size() == 1
def save_on_master(*opt, **kwopt):
    if is_main_process():
        torch.save(*opt, **kwopt)
        
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
        
def init_distributed_mode(opt):


    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ["RANK"])
        opt.world_size = int(os.environ['WORLD_SIZE']) # 1
        opt.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        opt.rank = int(os.environ['SLURM_PROCID'])
        opt.gpu = opt.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        opt.distributed = False
        return
    opt.distributed = True
    torch.cuda.set_device(opt.gpu)
    opt.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        opt.rank, opt.dist_url), flush=True)
    torch.distributed.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                         world_size=opt.world_size, rank=opt.rank)
    torch.distributed.barrier()
    setup_for_distributed(opt.rank == 0)
    print(opt)      
    
    
