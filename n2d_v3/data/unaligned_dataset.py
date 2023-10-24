import os.path, glob
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import json

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)

        datapath = os.path.join(opt.dataroot, opt.phase, '*')  # root/phase/A, root/phase/B # phase determines whether train/ test
        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs] # [A, B] 
        self.sizes = [len(p) for p in self.paths] # count for size of the images. A, B

    def load_image(self, dom, idx): 
        # A, B -> 0 , 1
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        img_A = self.load_image(0,index%self.sizes[0])
    
        if not self.opt.isTrain: # test not shuffling
            img_B = self.load_image(1,index%self.sizes[1])
        else: # shuffle when training.
            img_B = self.load_image(1,random.randint(0,self.sizes[1]-1))  
        
        return img_A, img_B

    def __len__(self): # return bigger file
        return max(self.sizes)
        
    def name(self):
        return 'UnalignedDataset'

class VFPDataset(BaseDataset):
    def __init__(self, opt):
        super(VFPDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)
        # change the dir
        json_file = '/root/jin/vfp290k/VFP290K_meta_data/light condition.json'
        
        vfp290k_dir = '/root/jin/vfp290k'
        with open(json_file) as f:
            json_object = json.load(f)
        self.data_path_day = self.parse_vfp290k(vfp290k_dir, json_object['day'])
        self.data_path_night = self.parse_vfp290k(vfp290k_dir, json_object['night'])
        self.sizes = [len(self.data_path_day), len(self.data_path_night)]
        self.path = [self.data_path_day, self.data_path_night]
        
    def parse_vfp290k(self, file_dir, json_object=None):
        data_path = []
        json_files = [ i for key in json_object.keys()
            for i in json_object[key]]
        json_file2 = '/root/jin/vfp290k/VFP290K_meta_data/background.json'
        with open(json_file2) as f:
            json_object2 = json.load(f)
        json_object2 = json_object2['building'] # to exclude
        json_files2 = [i for key in json_object2.keys() for i in json_object2[key]]
        for files in os.listdir(file_dir):
            if files.startswith('G'):
                if files in json_files:
                    if files not in json_files2:
                        data_path += (glob.glob(os.path.join(file_dir, files,'images')+'/*.jpg'))
        return data_path

    def load_image(self, dom, idx): 
        # A, B -> 0 , 1
        path = self.path[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        img_A = self.load_image(0, index%self.sizes[0])
    
        if not self.opt.isTrain: # test not shuffling
            img_B = self.load_image(1, index%self.sizes[1])
        else: # shuffle when training.
            img_B = self.load_image(1, random.randint(0,self.sizes[1]-1))  
        
        return img_A, img_B

    def __len__(self): # return bigger file
        return max(self.sizes)
        
    def name(self):
        return 'VFP290k_dataset'

