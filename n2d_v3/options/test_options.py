from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)        
        self.test_opt = self.parser.parse_args()
        self.isTrain = False

        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc (determines name of folder to load from)')
        self.parser.add_argument('--start_epoch', type=int, default=0, help='which epoch to load if continuing training, set 0 for new')
        ### display ###
        
        if self.test_opt.model_name == 'cycleGAN':
        
            self.parser.add_argument('--pool_size', type=int, default=50, help='pool size for ImagePool')

            self.parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator')
            self.parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator')

            self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam b1')
            self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam b2')
            
        elif self.test_opt.model_name == 'Adain':
    
            self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for discriminator')
            self.parser.add_argument('--alpha', type=float, default=0.0, help='instance norm sensitivity')
            
            self.parser.add_argument('--content_weight', type=float, default=1.0, help='content weight value')
            self.parser.add_argument('--style_weight', type=float, default=10.0, help='style weight value')

            self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam b1')
            self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam b2')
