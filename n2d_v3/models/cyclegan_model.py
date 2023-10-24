import numpy as np
import torch
import torch.nn as nn
import functools, itertools

from collections import OrderedDict
from itertools import chain
import utils.util as util
from utils.ddp import *
from utils.image_pool import ImagePool

## style loss 
import torchvision.models.vgg as vgg
from torch.autograd import Variable
from collections import namedtuple
import wandb
import copy
   

class Cyclegan():
    def name(self):
        return 'Cyclegan'

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.losses = {'G_A_losses':[], 'G_B_losses':[],'D_A_losses':[],'D_B_losses':[],'Cycle_A_losses':[], 'Cycle_B_losses':[]}
        self.p = {'p_D_A_real':[], 'p_D_A_fake':[],'p_D_B_real':[],'p_D_B_fake':[]}
        self.wandb = self.opt.wandb


        # load/define networks
        print('---------- Networks initialized -------------')
        self.model_path = os.path.join(self.save_dir, "weight_{}.pt".format(opt.start_epoch))
        
        self.netG_A, self.netG_B, self.netD_A, self.netD_B, self.G_optimizer, self.D_optimizer = define_cyclegan(opt, self.model_path)
        
        if opt.start_epoch > 0:
            print('model is loaded from checkpoint dir: {}'.format(self.model_path))
        # later make optional for style loss
        if opt.isTrain:
            print('Training configuration is initialized..')
            self.MSE_Loss, self.L1_Loss, self.loss_network = loss_setup(opt)
            self.fake_A_pool = ImagePool(opt)
            self.fake_B_pool = ImagePool(opt)
        print(self.netG_A)
        print('-----------------------------------------------')
        
        # device settings
        if opt.distributed:
            self.device = get_rank()
        else:
            self.device = opt.gpu_ids[0]
            
    def update_learning_rate(self, epoch):
        if (epoch) > self.opt.decay_epoch: 
            self.D_optimizer.param_groups[0]['lr'] -= self.opt.lrD / (self.opt.n_epochs - self.opt.decay_epoch)
            self.G_optimizer.param_groups[0]['lr'] -= self.opt.lrG / (self.opt.n_epochs - self.opt.decay_epoch)
            print('New learning rate D_optimizer: {}, G_optimizer: {}'.format(self.D_optimizer.param_groups[0]['lr'], self.G_optimizer.param_groups[0]['lr']))
    
    def initialize_training(self):
        # initialize all for training
        self.losses = {'G_A_losses':[], 'G_B_losses':[],'D_A_losses':[],'D_B_losses':[],'Cycle_A_losses':[], 'Cycle_B_losses':[]}
        self.p = {'p_D_A_real':[], 'p_D_A_fake':[],'p_D_B_real':[],'p_D_B_fake':[]}
        self.netG_A.train(), self.netG_B.train(), self.netD_A.train(), self.netD_B.train()
    
    def run_batch(self, epoch, i, dataset, data):
        self.set_input(data)
        self.optimize_parameters()
        if i % self.opt.display_freq == 0 and i > 1:
            self.get_current_losses(epoch, i, len(dataset),  epoch_vis=False, test_dataset=None)

    def set_input(self, input):
        
        real_A, real_B = input[0].cuda(self.device), input[1].cuda(self.device)
        
        # train generator G #
        fake_B = self.netG_A(real_A)
        D_B_fake_decision = self.netD_B(fake_B)
            
        G_A_loss = self.MSE_Loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda(self.device)))
        # forward cycle
        recon_A = self.netG_B(fake_B) # B -> A
        cycle_A_loss = self.L1_Loss(recon_A, real_A) * 10            
        styleA_loss = compStyle(real_A,fake_B, self.loss_network, self.MSE_Loss)  # 실제 A의 이미지, fake_b 이미지 | 실제 a이미지와 fake_b의 이미지간의 스타일이 멀어지지 않도록 유도.
        fake_A = self.netG_B(real_B)
       
        D_A_fake_decision = self.netD_A(fake_A)
        G_B_loss = self.MSE_Loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size()).cuda(self.device)))

        # backward cycle loss
        recon_B = self.netG_A(fake_A)
        cycle_B_loss = self.L1_Loss(recon_B, real_B) * 10        
        styleB_loss = compStyle(real_B,fake_A, self.loss_network, self.MSE_Loss) # 실제 이미지 B와 G_B가 생성한 A의 이미지간 스타일이 너무 멀어지지 않도록 유도함.
        
        style_loss = (styleA_loss + styleB_loss) 
    
        G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss 
        
        self.G_loss = G_loss+style_loss * 2.5 # style loss 가중치 2.5


         # -------------------------- train discriminator D_A --------------------------
        D_A_real_decision = self.netD_A(real_A)
        D_A_real_loss = self.MSE_Loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size()).cuda(self.device)))
        fake_A_D = self.fake_A_pool.query(fake_A)
        D_A_fake_decision = self.netD_A(fake_A_D)
        D_A_fake_loss = self.MSE_Loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size()).cuda(self.device)))   
        # Back propagation
        D_A_loss = (D_A_real_loss + D_A_fake_loss ) * 0.5
        # -------------------------- train discriminator D_B --------------------------
        D_B_real_decision = self.netD_B(real_B)
        D_B_real_loss = self.MSE_Loss(D_B_real_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda(self.device)))
        fake_B_D = self.fake_B_pool.query(fake_B)            
        D_B_fake_decision = self.netD_B(fake_B_D)
        D_B_fake_loss = self.MSE_Loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).cuda(self.device)))
        D_B_loss = (D_B_real_loss + D_B_fake_loss ) * 0.5
        
        
        self.D_loss = D_A_loss + D_B_loss
        # update the losses
        self.losses['G_A_losses'].append(G_A_loss.item()), self.losses['G_B_losses'].append(G_B_loss.item())
        self.losses['D_A_losses'].append(D_A_loss.item()), self.losses['D_B_losses'].append(D_B_loss.item())
        self.losses['Cycle_A_losses'].append(cycle_A_loss.item()), self.losses['Cycle_B_losses'].append(cycle_B_loss.item())
        # for wandb
        self.real_A = real_A
        self.real_B = real_B
        self.fake_A = fake_A
        self.fake_B = fake_B
        self.recon_A = recon_A
        self.recon_B = recon_B
        
    def optimize_parameters(self):
        # G_A and G_B
        self.G_optimizer.zero_grad()
        self.G_loss.backward()
        self.G_optimizer.step()
        # D_A and D_B
        self.D_optimizer.zero_grad()
        self.D_loss.backward()
        self.D_optimizer.step()
        
    def get_current_losses(self, epoch, step, total_step, epoch_vis=False, test_dataset=None):
        self.test_dataset = test_dataset
        self.epoch_vis = epoch_vis
        if epoch_vis: # for epoch
            avg_G_A_losses, avg_G_B_losses, avg_D_A_losses, avg_D_B_losses, avg_Cycle_A_losses, avg_Cycle_B_losses = [sum(self.losses[key])/len(self.losses[key]) for key in self.losses.keys()]
            if self.opt.distributed:
                if self.device==0:
                    print('Epoch [%d/%d], avg_D_A_loss: %.4f, avg_D_B_loss: %.4f, avg_G_A_loss: %.4f, avg_G_B_loss: %.4f, avg_Cycle_A_losses: %.5f, avg_Cycle_B_losses: %.5f' % 
                    (epoch+1, self.opt.n_epochs, avg_D_A_losses, avg_D_B_losses, avg_G_A_losses, avg_G_B_losses, avg_Cycle_A_losses, avg_Cycle_B_losses))

                    if test_dataset != None:
                        self.test(test_dataset)
                    if self.wandb:
                        self.wandb_visualization(epoch, avg_G_A_losses, avg_G_B_losses, avg_D_A_losses, avg_D_B_losses, avg_Cycle_A_losses, avg_Cycle_B_losses)
            else:
                print('Epoch [%d/%d], avg_D_A_loss: %.4f, avg_D_B_loss: %.4f, avg_G_A_loss: %.4f, avg_G_B_loss: %.4f, avg_Cycle_A_losses: %.5f, avg_Cycle_B_losses: %.5f' % 
                (epoch+1, self.opt.n_epochs, avg_D_A_losses, avg_D_B_losses, avg_G_A_losses, avg_G_B_losses, avg_Cycle_A_losses, avg_Cycle_B_losses))
                
                if test_dataset != None:
                    self.test(test_dataset)
                if self.wandb:
                    self.wandb_visualization(epoch, avg_G_A_losses, avg_G_B_losses, avg_D_A_losses, avg_D_B_losses, avg_Cycle_A_losses, avg_Cycle_B_losses)

        else: # for batch 
            G_A_losses, G_B_losses, D_A_losses, D_B_losses, Cycle_A_losses, Cycle_B_losses = [self.losses[key][step] for key in self.losses.keys()]
            
            if self.opt.distributed:
                if self.device==0:
                    print('Epoch [%d/%d], Step [%d/%d], D_A_loss: %.4f, D_B_loss: %.4f, G_A_loss: %.4f, G_B_loss: %.4f, Cycle_A_losses: %.5f, Cycle_B_losses: %.5f' % 
                    (epoch+1, self.opt.n_epochs, step+1, total_step, D_A_losses, D_B_losses, G_A_losses, G_B_losses, Cycle_A_losses, Cycle_B_losses))
                    if self.wandb:
                        self.wandb_visualization(epoch, G_A_losses, G_B_losses, D_A_losses, D_B_losses, Cycle_A_losses, Cycle_B_losses)

            else:
                print('Epoch [%d/%d], Step [%d/%d], D_A_loss: %.4f, D_B_loss: %.4f, G_A_loss: %.4f, G_B_loss: %.4f, Cycle_A_losses: %.5f, Cycle_B_losses: %.5f' % 
                (epoch+1, self.opt.n_epochs, step+1, total_step, D_A_losses, D_B_losses, G_A_losses, G_B_losses, Cycle_A_losses, Cycle_B_losses))
                if self.wandb:
                    self.wandb_visualization(epoch, G_A_losses, G_B_losses, D_A_losses, D_B_losses, Cycle_A_losses, Cycle_B_losses)
                    
    def wandb_visualization(self, epoch, G_A_losses, G_B_losses, D_A_losses, D_B_losses, Cycle_A_losses, Cycle_B_losses):
        if self.epoch_vis:

            wandb.log(
                        {           "Epoch": epoch+1,
                                    "Epoch_Loss_G(X to Y)": round(G_A_losses, 4),
                                    "Epoch_Loss_F(Y to X)": round(G_B_losses, 4),
                                    "Epoch_StyleA Loss": round(Cycle_A_losses, 4),
                                    "Epoch_StyleB Loss": round(Cycle_B_losses, 4),
                                    "Epoch_Loss_D(X to Y)": round(D_A_losses, 4),
                                    "Epoch_Loss_D(Y to X)": round(D_B_losses, 4),
                        }
                    )
            if self.test_dataset:
                p_D_A_real, p_D_A_fake, p_D_B_real, p_D_B_fake = [self.p[key][0] for key in self.p.keys()]
                print("Epoch: {} p_D_A_real: {} p_D_A_fake: {} p_D_B_real: {} p_D_B_fake: {}".format(epoch+1, p_D_A_real, p_D_A_fake, p_D_B_real, p_D_B_fake))
                wandb.log(
                {   "Epoch": epoch+1,
                    "p_D_A_real": round(p_D_A_real, 4),
                    "p_D_A_fake": round(p_D_A_fake, 4),
                    "p_D_B_real": round(p_D_B_real, 4),
                    "p_D_B_fake": round(p_D_B_fake, 4),
                })

        else:
            wandb.log(
                        {           "Epoch": epoch+1,
                                    "Loss_G(X to Y)": round(G_A_losses, 4),
                                    "Loss_F(Y to X)": round(G_B_losses, 4),
                                    "StyleA Loss": round(Cycle_A_losses, 4),
                                    "StyleB Loss": round(Cycle_B_losses, 4),
                                    "Loss_D(X to Y)": round(D_A_losses, 4),
                                    "Loss_D(Y to X)": round(D_B_losses, 4)
                        }
                    )
            
            wandb.log(
                            {
                                "X": wandb.Image(util.denormalize_image(self.real_A[0].clone().detach().cpu())),
                                "Y": wandb.Image(util.denormalize_image(self.real_B[0].clone().detach().cpu())),
                                "Generated Target (X->Y)": wandb.Image(util.denormalize_image(self.fake_B[0].clone().detach().cpu())), # G(X)
                                "Reconstructed Target (X->Y->X)": wandb.Image(util.denormalize_image(self.recon_A[0].clone().detach().cpu())),
                                "Generated Input (Y->X)": wandb.Image(util.denormalize_image(self.fake_A[0].clone().detach().cpu())), # F(Y)
                                "Reconstructed Input (Y->X->Y)": wandb.Image(util.denormalize_image(self.recon_B[0].clone().detach().cpu()))
                            })

    def model_save(self, epoch):
        self.model_path = os.path.join(self.save_dir, "weight_{}.pt".format(epoch))
        
        print('model is saved from checkpoint dir: {}'.format(self.model_path))
        try:
            state_dict1 = self.netG_A.module.state_dict()
            state_dict2 = self.netG_B.module.state_dict()        
            state_dict3= self.netD_A.module.state_dict()
            state_dict4 = self.netD_B.module.state_dict()
            G_optimizer = self.G_optimizer.state_dict()
            D_optimizer = self.D_optimizer.state_dict()
        except AttributeError:
            state_dict1 = self.netG_A.state_dict()
            state_dict2 = self.netG_B.state_dict()        
            state_dict3= self.netD_A.module.state_dict()
            state_dict4 = self.netD_B.module.state_dict()
            G_optimizer = self.G_optimizer.state_dict()
            D_optimizer = self.D_optimizer.state_dict()
        
        torch.save({
                'G_A_state_dict': state_dict1,
                'G_B_state_dict': state_dict2,
                'D_A_state_dict': state_dict3,
                'D_B_state_dict': state_dict4,
                'G_optimizer_state_dict' : G_optimizer,
                'D_optimizer_state_dict' : D_optimizer,
                }, self.model_path)

    def test(self,test_dataset):
        
        print('---------- Testing the model -------------')
        
        self.netG_A.eval(), self.netG_B.eval(), self.netD_A.eval(), self.netD_B.eval()
        p_D_A_real, p_D_A_fake, p_D_B_real, p_D_B_fake = 0.,0., 0., 0.
        with torch.autograd.no_grad():
            for i, (real_A, real_B) in enumerate(test_dataset):
                
                real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                    
                fake_B = self.netG_A(real_A)
                fake_A = self.netG_B(real_B)            

                D_B_fake_decision = torch.mean(self.netD_B(fake_B)) # 확률 진짠지 가짠지. LSGAN loss 활용하여 
                D_A_fake_decision = torch.mean(self.netD_A(fake_A))

                p_D_A_real += (torch.mean(self.netD_A(real_A)).item())/len(test_dataset)
                p_D_A_fake += ((D_A_fake_decision).item())/len(test_dataset)
                
                p_D_B_real += (torch.mean(self.netD_B(real_B)).item())/len(test_dataset)
                p_D_B_fake += ((D_B_fake_decision).item())/len(test_dataset)

        self.p['p_D_A_real'].append(p_D_A_real), self.p['p_D_A_fake'].append(p_D_A_fake)
        self.p['p_D_B_real'].append(p_D_B_real), self.p['p_D_B_fake'].append(p_D_B_fake)
        
    def run_test(self, i, data, save=False):
        # A -> B only.
        self.netG_A.eval()
        real_A, real_B = data[0].cuda(self.device), data[1].cuda(self.device) # real_A = content_img, real_B = style_img
        
        output = self.netG_A(real_A)

        output = np.transpose(util.denormalize_image(output.clone().detach().cpu().numpy()), (1,2,0))
        if save:
            img_save_dir =os.path.join(self.save_dir, 'test_imgs')
            util.mkdir(img_save_dir)
            util.save_imgs(img_save_dir, i, output)


def define_cyclegan(opt, path): 
    G_A, G_B, D_A, D_B = define_G(), define_G(), define_D(), define_D()
    G_optimizer, D_optimizer = optimizer_setup(G_A, G_B, D_A, D_B, opt)
    
    if opt.start_epoch != 0: # load from checkpoint
        G_A, G_B, D_A, D_B, go, do = model_load(G_A, G_B, D_A, D_B, G_optimizer, D_optimizer, path)
        G_optimizer, D_optimizer = optimizer_setup(G_A, G_B, D_A, D_B, opt) # prevent different device allocation

    if len(opt.gpu_ids) > 0:
        G_A, G_B, D_A, D_B = model_gpu_setup(G_A,opt), model_gpu_setup(G_B,opt), model_gpu_setup(D_A,opt), model_gpu_setup(D_B,opt)
    return G_A, G_B, D_A, D_B, G_optimizer, D_optimizer


def model_load(G_A, G_B, D_A, D_B, G_optimizer, D_optimizer, PATH):
    checkpoint = torch.load(PATH)
    
    G_A.load_state_dict(checkpoint['G_A_state_dict'])  # load model state dict
    G_B.load_state_dict(checkpoint['G_B_state_dict'])  # load model state dict
    D_A.load_state_dict(checkpoint['D_A_state_dict'])  # load model state dict
    D_B.load_state_dict(checkpoint['D_B_state_dict'])  # load model state dict

    G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])  # load optim state dict
    D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])  # load optim state dict

    return G_A, G_B, D_A, D_B, G_optimizer, D_optimizer


def optimizer_setup(G_A, G_B, D_A, D_B, opt):  # 이것도 pix2pix 보고 나중에 손보기
    G_optimizer = torch.optim.Adam(chain(G_A.parameters(), G_B.parameters()), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
    D_optimizer = torch.optim.Adam(chain(D_A.parameters(), D_B.parameters()), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
    
    return G_optimizer, D_optimizer

def model_gpu_setup(m,opt):
    if len(opt.gpu_ids) == 1:
        assert(torch.cuda.is_available())
        m.cuda(opt.gpu_ids[0])
    elif len(opt.gpu_ids) > 1: 
        if opt.distributed:
            m.cuda(get_rank())
            m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
            m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[get_rank()], find_unused_parameters=False)
        else:
            m = nn.DataParallel(m).cuda(opt.gpu_ids[0])
    return m

def define_G(input_nc=3, n_blocks=9, upsampling=True, norm='instance', use_dropout=False):
    
    G = Generator(input_nc, n_residual_blocks=n_blocks, upsampling=True, norm_type=norm, use_dropout=use_dropout)    
    G.apply(weights_init)
            
    return G

def define_D(input_nc=3):
    
    D = Discriminator(input_nc)
    D.apply(weights_init)

    return D

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

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

### model architecture ###

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect'):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim , dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        output = self.conv_block(x) + x
        return output

class Generator(nn.Module):
    # for n_residual_blocks use 6 for small size image (256x256), 9 for large (512x512)
    def __init__(self, input_nc=3, n_residual_blocks=6, use_dropout=False, use_bias=False, norm_type = 'instance', upsampling=False):
        super(Generator, self).__init__()
        self.upsampling = upsampling
        self.norm_layer = get_norm_layer(norm_type)
        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResnetBlock(in_features, self.norm_layer, use_dropout, use_bias)]
        # Upsampling
        out_features = in_features//2
        
        for _ in range(2):
            if self.upsampling:
                    model +=[nn.Upsample(scale_factor = 2, mode='bilinear'),
                                nn.ReflectionPad2d(1),    
                                nn.Conv2d(in_features, out_features,
                                        kernel_size=3, stride=1, padding=0),
                                nn.InstanceNorm2d(out_features),
                                nn.ReLU(inplace=True)] 
            else:
                model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                            nn.InstanceNorm2d(out_features),
                            nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, input_nc, 7), # input_nc = output_nc
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x) 
        return x
       # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
### style loss and loss

def loss_setup(opt):
    if opt.distributed:
        MSE_Loss = torch.nn.MSELoss().cuda(get_rank())
        L1_Loss = torch.nn.L1Loss().cuda(get_rank())
        loss_network = LossNetwork(opt)
    else:
        device = opt.gpu_ids[0]
        MSE_Loss = torch.nn.MSELoss().cuda(device)
        L1_Loss = torch.nn.L1Loss().cuda(device)
        loss_network = LossNetwork(opt)
    loss_network.eval()
        
    return MSE_Loss, L1_Loss, loss_network

class LossNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(LossNetwork, self).__init__()
        if len(opt.gpu_ids) > 0:
            if opt.distributed:
                self.device = get_rank()
            else:
                self.device = opt.gpu_ids[0]
        self.LossOutput =  namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        self.vgg_layers = vgg.vgg16(weights='VGG16_Weights.DEFAULT').cuda(self.device)

        
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
                output[self.layer_name_mapping[name]] = x
        return self.LossOutput(**output)



def gram_matrix(y): # https://aigong.tistory.com/360   --> check this out!
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def compStyle(a,b, loss_network, MSE_Loss): 
    #http://pytorch.org/docs/master/notes/autograd.html#volatile
    with torch.no_grad():
        styleB_loss_features = loss_network(Variable(a)) # features are extracted from pretrained vgg16 from different layers
    gram_style = [Variable(gram_matrix(y).data, requires_grad=False) for y in styleB_loss_features] # 각 feature에 대한 gram matrix 연산
    # 즉 a 이미지로 부터 뽑힌 feature의 gram_matrix 연산을 통해 각 channel x channel 의 matrix를 얻고
    # 이 matrix는 feature들로부터의 스타일을 의미한다.
        
    features_y = loss_network(b) # 마찬가지로 b에서도 feature를 뽑고.
        
    style_loss = 0    
    for m in range(len(features_y)):
        gram_s = gram_style[m]
        gram_y = gram_matrix(features_y[m])
        style_loss += 1e4 * MSE_Loss(gram_y, gram_s.expand_as(gram_y)) # style loss는 즉, 이미지 a 와 b
    # style loss는 즉 이미지 a 와 b로 부터 pretrained된 vgg net을 활용하여 feature를 extract 한뒤
    # extract 된 feature에 대하여 gram_matrix를 연산해 주게된다.
    # 이후, 각각 연산된 gram_matrix의 거리 차를 1000에 곱해주는것.
    # 즉 이미지 a와 b의 거리가 같다면 if a==b, style_loss = 0.
    return style_loss
