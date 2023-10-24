import numpy as np
import torch
import torch.nn as nn
import functools, itertools

from collections import OrderedDict
from itertools import chain
import utils.util as util
from utils.ddp import *

## style loss 
import torchvision.models.vgg as vgg
from torch.autograd import Variable
from collections import namedtuple
import wandb
from tqdm import tqdm


def run_batch(epoch, i, dataset, data, model, opt):
    model.set_input(data)
    model.optimize_parameters()
    if i % opt.display_freq == 0 and i > 1:
        model.get_current_losses(epoch, i, len(dataset),  epoch_vis=False, test_dataset=None)

class Adain():
    def name(self):
        return 'Adain'

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.losses = {'Style_losses':[], 'Content_losses':[]}
        self.wandb = self.opt.wandb


        # load/define networks
        print('---------- Networks initialized -------------')
        self.model_path = os.path.join(self.save_dir, "weight_{}.pt".format(opt.start_epoch))
        self.adain_decoder, self.adain_optimizer = define_adain(opt, self.model_path)
 
        if opt.start_epoch > 0:
            print('model is loaded from checkpoint dir: {}'.format(self.model_path))

        # later make optional for style loss
        self.MSE_Loss, self.feature_network = loss_setup(opt)
        print(self.adain_decoder)
        print('-----------------------------------------------')
        
        # device settings
        if opt.distributed:
            self.device = get_rank()
        else:
            self.device = opt.gpu_ids[0]
            
    def update_learning_rate(self, epoch):
        if (epoch) > self.opt.decay_epoch: 
            self.adain_optimizer.param_groups[0]['lr'] -= self.opt.lr / (self.opt.n_epochs - self.opt.decay_epoch)
            print('New learning rate adain optimizer: {}'.format(self.adain_optimizer.param_groups[0]['lr']))
    
    def initialize_training(self):
        # initialize all for training
        self.losses = {'Style_losses':[], 'Content_losses':[]}
        self.adain_decoder.train()
        
    def set_input(self, input):
        
        content, style = input[0].cuda(self.device), input[1].cuda(self.device) # real_A = content_img, real_B = style_img
        
        content_features = self.feature_network(content)
        style_features = self.feature_network(style)

        t = adaptive_instance_normalization(content_features['relu4_3'], style_features['relu4_3']) # t를 구함. 마지막 레이어 값으로.
        t = self.opt.alpha* t + (1-self.opt.alpha) * content_features['relu4_3']
        
        g_t = self.adain_decoder(t)
        g_t_feats = self.feature_network(g_t)        

        loss_c = self.MSE_Loss(g_t_feats['relu4_3'], t) # decoder로 sytle 변환된 이미지에 vgg feature - normalize된 content image
        loss_s = 0

        keys = list(g_t_feats.keys())
        for i in range(0, 4):
            loss_s += calc_style_loss(g_t_feats[keys[i]], style_features[keys[i]], self.MSE_Loss)

        self.loss_c = self.opt.content_weight * loss_c
        self.loss_s = self.opt.style_weight * loss_s

        self.loss = loss_c + loss_s
  
        # update the losses
        self.losses['Content_losses'].append(loss_c.item()), self.losses['Style_losses'].append(loss_s.item())
        # for wandb
        self.content = content
        self.style = style
        self.g_t = g_t
        
    def optimize_parameters(self):
        # G_A and G_B
        self.adain_optimizer.zero_grad()
        self.loss.backward()
        self.adain_optimizer.step()
    
    def get_current_losses(self, epoch, step, total_step, epoch_vis=False, test_dataset=None):
        self.epoch_vis = epoch_vis
        if epoch_vis: # for epoch
            avg_style_loss, avg_content_loss = [sum(self.losses[key])/len(self.losses[key]) for key in self.losses.keys()]
            if self.opt.distributed:
                if self.device==0:
                    print('Epoch [%d/%d], avg_content_loss: %.4f, avg_style_loss: %.4f' % 
                    (epoch+1, self.opt.n_epochs, avg_content_loss, avg_style_loss))
                
                    if self.wandb:
                        self.wandb_visualization(epoch, avg_style_loss, avg_content_loss)
            else:
                print('Epoch [%d/%d], avg_content_loss: %.4f, avg_style_loss: %.4f' % 
                    (epoch+1, self.opt.n_epochs, avg_content_loss, avg_style_loss))
                
                if self.wandb:
                    self.wandb_visualization(epoch, avg_style_loss, avg_content_loss)

        else: # for batch 
            style_losses, content_losses = [self.losses[key][step] for key in self.losses.keys()]
            
            if self.opt.distributed:
                if self.device==0:
                    print('Epoch [%d/%d], Step [%d/%d], content_loss: %.4f, style_loss: %.4f' % 
                    (epoch+1, self.opt.n_epochs, step+1, total_step, content_losses, style_losses))
                    if self.wandb:
                        self.wandb_visualization(epoch, style_losses, content_losses)

            else:
                print('Epoch [%d/%d], Step [%d/%d], content_loss: %.4f, style_loss: %.4f' % 
                    (epoch+1, self.opt.n_epochs, step+1, total_step, content_losses, style_losses))
                if self.wandb:
                    self.wandb_visualization(epoch, style_losses, content_losses)
                    
    def wandb_visualization(self, epoch, style_losses, content_losses):
        if self.epoch_vis:

            wandb.log(
                        {           "Epoch": epoch+1,
                                    "Epoch_Style_loss": round(style_losses, 4),
                                    "Epoch_Content_loss": round(content_losses, 4),
                        }
                    )
        else:
            wandb.log(
                        {           "Epoch": epoch+1,
                                    "Style_loss": round(style_losses, 4),
                                    "Content_loss": round(content_losses, 4),
                        }
                    )
            
            wandb.log(
                            {
                                "style": wandb.Image(util.denormalize_image(self.style[0].clone().detach().cpu())),
                                "content": wandb.Image(util.denormalize_image(self.content[0].clone().detach().cpu())),
                                "content -> style ": wandb.Image(util.denormalize_image(self.g_t[0].clone().detach().cpu())), # G(X)
                            })

    def model_save(self, epoch):
        self.model_path = os.path.join(self.save_dir, "weight_{}.pt".format(epoch))
        
        print('model is saved from checkpoint dir: {}'.format(self.model_path))
        try:
            state_dict1 = self.adain_decoder.module.state_dict()
            adain_optimizer = self.adain_optimizer.state_dict()
        except AttributeError:
            state_dict1 = self.adain_decoder.module.state_dict()
            adain_optimizer = self.adain_optimizer.state_dict()

        torch.save({
                'adain_decoder': state_dict1,
                'adain_optimizer' : adain_optimizer,
                }, self.model_path)
    
    def run_test(self, i, data, save=False):
        # t of test samples, what is test samples are not given? --> need to think
        # only one reference image is used in adain paper.
        # average? -> need to consider
        feature_network = self.feature_network
        decoder = self.adain_decoder
        
        decoder.eval()
            
        content, style = data[0].cuda(self.device), data[1].cuda(self.device) # real_A = content_img, real_B = style_img
        fixed_style = None # need to consider.
        # average? fixed_style? raw?
        content_features = feature_network(content)
        style_features = feature_network(style)
        
        t = adaptive_instance_normalization(content_features['relu4_3'], style_features['relu4_3'])
        t = self.opt.alpha* t + (1-self.opt.alpha) * content_features['relu4_3']
        output = decoder(t)
        
        output = np.transpose(util.denormalize_image(output.clone().detach().cpu().numpy()), (1,2,0))
        if save:
            img_save_dir =os.path.join(self.save_dir, 'test_imgs')
            util.mkdir(img_save_dir)
            util.save_imgs(img_save_dir, i, output)
        

def model_load(adain_decoder, adain_optimizer, PATH):
    checkpoint = torch.load(PATH)
    
    adain_decoder.load_state_dict(checkpoint['adain_decoder'])  # load model state dict
    adain_optimizer.load_state_dict(checkpoint['adain_optimizer'])  # load optim state dict

    return adain_decoder, adain_optimizer

def model_gpu_setup(m,opt):
    if len(opt.gpu_ids) == 1:
        assert (torch.cuda.is_available())
        m.cuda(opt.gpu_ids[0])
    elif len(opt.gpu_ids) > 1: 
        if opt.distributed:
            m.cuda(get_rank())
            m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
            m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[get_rank()], find_unused_parameters=False)
        else:
            m = nn.DataParallel(m).cuda(opt.gpu_ids[0])
    return m

def define_adain(opt, path):  # rename
    adain_decoder = define_decoder()    
    adain_optimizer = optimizer_setup(adain_decoder, opt)
    
    if opt.start_epoch != 0: # load from checkpoint
        adain_decoder, adain_optimizer = model_load(adain_decoder, adain_optimizer, path)
    
    if len(opt.gpu_ids) > 0:
        adain_decoder = model_gpu_setup(adain_decoder,opt)
    

    return adain_decoder, adain_optimizer

def optimizer_setup(adain_decoder, opt):  # 이것도 pix2pix 보고 나중에 손보기
    adain_optimizer = torch.optim.Adam(adain_decoder.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    
    return adain_optimizer


def define_decoder():
    # weight initialization
    
    adain_decoder = Adain_decoder()    
    adain_decoder.apply(weights_init)
            
    return adain_decoder

def loss_setup(opt):
    if opt.distributed:
        MSE_Loss = torch.nn.MSELoss().cuda(get_rank())
        feature_network = featureNetwork().cuda(get_rank())
    else:
        device = opt.gpu_ids[0]
        MSE_Loss = torch.nn.MSELoss().cuda(device)
        feature_network = featureNetwork().cuda(device)
    feature_network.eval()
        
    return MSE_Loss, feature_network

### models ###
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

### model architecture ###
# for adain, only the decoder is trained

import torchvision.models.vgg as vgg

class featureNetwork(torch.nn.Module):
    def __init__(self): # 3,256,256
        super(featureNetwork, self).__init__()
        self.vgg_layers = vgg.vgg16(weights='VGG16_Weights.DEFAULT')
        
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x # extract the features from designated layers of VGG16.
        return output

class Adain_decoder(nn.Module):
    def __init__(self):
        super(Adain_decoder, self).__init__() # input -> batch, (512, 32, 32)
        
        
        model = [nn.ReflectionPad2d((1,1,1,1)),
                 nn.Conv2d(512, 256, (3,3)),
                 nn.ReLU()]
    
        conv_layers = [4, 2, 2]
        fs = 256
        for j, layers in enumerate(conv_layers):
            model += [nn.Upsample(scale_factor=2, mode='nearest')]
            for i in range(layers):
                if i== layers-1 and j == len(conv_layers)-1:
                    model += [nn.ReflectionPad2d((1,1,1,1)),
                            nn.Conv2d(fs, 3, 3),
                            nn.ReLU()]
                elif i == layers-1:
                    model += [nn.ReflectionPad2d((1,1,1,1)),
                            nn.Conv2d(fs, fs//2, 3),
                            nn.ReLU()]
                    fs = fs//2
                else:
                    model +=  [nn.ReflectionPad2d((1,1,1,1)),
                        nn.Conv2d(fs, fs, 3),
                        nn.ReLU()]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    
def calc_style_loss(input, target, mse_loss):
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    
    return mse_loss(input_mean, target_mean) + mse_loss(input_std, target_std)

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat) 
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size) # normalized feature of content
    return normalized_feat * style_std.expand(size) + style_mean.expand(size) # t
    
def calc_mean_std(feat, eps=1e-5): # 각 채널에 대해 variance 및 mean 연산. ex) mean, variance represented by one value. 
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4) # size needs to be (batch, channel, width, height)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
