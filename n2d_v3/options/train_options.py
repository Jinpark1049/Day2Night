from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True

        self.train_opt = self.parser.parse_args()
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc (determines name of folder to load from)')
        self.parser.add_argument('--start_epoch', type=int, default=0, help='which epoch to load if continuing training, set 0 for new')
        self.parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
        self.parser.add_argument('--decay_epoch', type=int, default=100, help='number of ecpohs for linearly decay learning rate')
        ### display ###
        self.parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')

        if self.train_opt.model_name == 'cycleGAN':
        
            self.parser.add_argument('--pool_size', type=int, default=50, help='pool size for ImagePool')

            self.parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator')
            self.parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator')

            self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam b1')
            self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam b2')
            
        elif self.train_opt.model_name == 'Adain':
    
            self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for discriminator')
            self.parser.add_argument('--alpha', type=float, default=0.0, help='instance norm sensitivity')
            
            self.parser.add_argument('--content_weight', type=float, default=1.0, help='content weight value')
            self.parser.add_argument('--style_weight', type=float, default=10.0, help='style weight value')

            self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam b1')
            self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam b2')

        

