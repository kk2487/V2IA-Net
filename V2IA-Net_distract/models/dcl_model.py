import itertools
import torch
from .base_model import BaseModel
#from . import networks
from . import networks_with_pretrain as networks
from .patchnce import PatchNCELoss
import util.util as util
from util.image_pool import ImagePool
import cv2
import numpy as np

def read_classes(file_path):

    fp = open(file_path, "r")
    classes = fp.readline()
    classes = classes. split(",")
    fp.close()

    return classes

classes = read_classes('classes.txt')

dn = ['distract', 'normal']

class DCLModel(BaseModel):
    """ This class implements DCLGAN model.
    This code is inspired by CUT and CycleGAN.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for DCLGAN """
        parser.add_argument('--DCL_mode', type=str, default="DCL", choices='DCL')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=2.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=1.0, help='weight for l1 identical loss: (G(X),X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization.")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for DCLGAN.
        if opt.DCL_mode.lower() == "dcl":
            parser.set_defaults(nce_idt=True, lambda_NCE=2.0)
        else:
            raise ValueError(opt.DCL_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'NCE1', 'D_B', 'G_B', 'NCE2', 'G', 'AR', 'DN']
        visual_names_A = ['real_A', 'fake_B']
        self.predict_names = ['predict_A', 'predict_A_dn']
        #visual_names_B = ['real_B', 'fake_A']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['idt_B', 'idt_A']
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        #self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        self.visual_names = visual_names_A
        if self.isTrain:
            self.model_names = ['G_A', 'F1', 'D_A', 'G_B', 'F2', 'D_B']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B']

        # define networks (both generator and discriminator)

        # 細節定義在networks.py裡面

        # 用來將A domain轉換成B domain
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        # 用來將B domain轉換成A domain
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        # 用來判別netG_A (B domain)轉換結果是否真實
        self.netF1 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)
        # 用來判別netG_B (A domain)轉換結果是否真實
        self.netF2 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions

            self.cnn_loss_function = torch.nn.CrossEntropyLoss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionSim = torch.nn.L1Loss('sum').to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_G_loss().backward()  # calculate graidents for G
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF1.parameters(), self.netF2.parameters()))
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        """
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.path_A = input['A_paths']   
        self.label_A = input['A_label' if AtoB else 'B_label'].to(device=self.device, dtype=torch.long)
        self.label_B = input['B_label' if AtoB else 'A_label'].to(device=self.device, dtype=torch.long)
        #print("-------A-------")
        if(self.label_A != 1):
            #print("not normal", self.label_A)
            self.label_A_dn = torch.tensor([0]).to(device=self.device, dtype=torch.long)
        else:
            #print("normal", self.label_A)
            self.label_A_dn = torch.tensor([1]).to(device=self.device, dtype=torch.long)
        #print("-------B-------")
        if(self.label_B != 1):
            #print("not normal", self.label_B)
            self.label_B_dn = torch.tensor([0]).to(device=self.device, dtype=torch.long)
        else:
            #print("normal", self.label_B)
            self.label_B_dn = torch.tensor([1]).to(device=self.device, dtype=torch.long)
        """
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.predict_A, self.predict_A_dn = self.netG_A(self.real_A)  # G_A(A)
        """
        self.fake_A, self.predict_B, self.predict_B_dn = self.netG_B(self.real_B)  # G_B(B)

        label = classes[self.label_A.cpu()]
        output = classes[self.predict_A.argmax()]
        label_dn = dn[self.label_A_dn.cpu()]
        out_dn = dn[self.predict_A_dn.argmax()]

        #print(output)
        A_np = self.real_A.squeeze(0).squeeze(0).cpu().numpy()
        #A_np = np.uint8(A_np)
        cv2.rectangle(A_np, (0, 0), (80, 20), 0, -1, cv2.LINE_AA)
        cv2.putText(A_np, label, (0, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255), 1)
        cv2.rectangle(A_np, (0, 20), (80, 40), 255, -1, cv2.LINE_AA)
        cv2.putText(A_np, output, (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0), 1)

        cv2.rectangle(A_np, (180, 0), (256, 20), 0, -1, cv2.LINE_AA)
        cv2.putText(A_np, label_dn, (180, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255), 1)
        cv2.rectangle(A_np, (180, 20), (256, 40), 255, -1, cv2.LINE_AA)
        cv2.putText(A_np, out_dn, (180, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0), 1)
        self.prdict_img = torch.from_numpy(A_np).float()
        self.prdict_img = torch.unsqueeze(self.prdict_img, dim = 0)
        self.prdict_img = torch.unsqueeze(self.prdict_img, dim = 0).to(self.device)
        """

        """
        cv2.imshow("tmp", A_np)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
        """

        """
        if self.opt.nce_idt:
            #print("--------idt")
            self.idt_A, self.predict_B_i, self.predict_B_i_dn = self.netG_A(self.real_B)
            self.idt_B, self.predict_A_i, self.predict_A_i_dn = self.netG_B(self.real_A)
        """
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) * self.opt.lambda_GAN

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * self.opt.lambda_GAN

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fakeB = self.fake_B
        fakeA = self.fake_A
        predict_A = self.predict_A  # G_A(A)
        predict_B = self.predict_B  # G_B(B)
        predict_B_i = self.predict_B_i  # G_B(A)
        predict_A_i = self.predict_A_i  # G_A(B)

        predict_A_dn = self.predict_A_dn
        predict_B_dn = self.predict_B_dn
        predict_B_i_dn = self.predict_B_i_dn  # G_B(A)
        predict_A_i_dn = self.predict_A_i_dn  # G_A(B)

        #print(predict_A.shape, self.label_A.shape)
        #print(predict_A)
        #print("-------------------------",self.label_A)
        #print(self.path_A)
        self.loss_AR = self.cnn_loss_function(predict_A, self.label_A) + self.cnn_loss_function(predict_B, self.label_B) \
                    + self.cnn_loss_function(predict_A_i, self.label_A) + self.cnn_loss_function(predict_B_i, self.label_B)

        self.loss_DN = self.cnn_loss_function(predict_A_dn, self.label_A_dn) + self.cnn_loss_function(predict_B_dn, self.label_B_dn) \
                    + self.cnn_loss_function(predict_A_i_dn, self.label_A_dn) + self.cnn_loss_function(predict_B_i_dn, self.label_B_dn)
        
        #print("----cnn_loss----", self.loss_AR)
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fakeB = self.netD_A(fakeB)
            pred_fakeA = self.netD_B(fakeA)
            self.loss_G_A = self.criterionGAN(pred_fakeB, True).mean() * self.opt.lambda_GAN
            self.loss_G_B = self.criterionGAN(pred_fakeA, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_A = 0.0
            self.loss_G_B = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE1 = self.calculate_NCE_loss1(self.real_A, self.fake_B) * self.opt.lambda_NCE
            self.loss_NCE2 = self.calculate_NCE_loss2(self.real_B, self.fake_A) * self.opt.lambda_NCE
        else:
            self.loss_NCE1, self.loss_NCE_bd, self.loss_NCE2 = 0.0, 0.0, 0.0
        if self.opt.lambda_NCE > 0.0:

            # L1 IDENTICAL Loss
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_IDT
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_IDT
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5 + (self.loss_idt_A + self.loss_idt_B) * 0.5

            # print("this it NCE loss")

        else:
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5

        self.loss_G = (self.loss_G_A + self.loss_G_B) * 0.2 + loss_NCE_both * 0.2 + self.loss_AR * 0.2 + self.loss_DN * 0.2
        #self.loss_G = (self.loss_G_A + self.loss_G_B) * 0.1 + loss_NCE_both * 0.1 + self.loss_AR * 0.8
        return self.loss_G

    def calculate_NCE_loss1(self, src, tgt):
        # src = real A
        # tgt = fake B
        n_layers = len(self.nce_layers)
        feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True) #拿取fake B特徵
        feat_k = self.netG_A(src, self.nce_layers, encode_only=True) #拿取real A特徵

        feat_k_pool, sample_ids = self.netF1(feat_k, self.opt.num_patches, None) #藉由判別器判斷real A特徵
        feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, sample_ids) #藉由判別器判斷fake B特徵
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_NCE_loss2(self, src, tgt):
        # src = real B
        # tgt = fake A
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True) #拿取fake A特徵
        feat_k = self.netG_B(src, self.nce_layers, encode_only=True) #拿取real B特徵
        feat_k_pool, sample_ids = self.netF2(feat_k, self.opt.num_patches, None) #藉由判別器判斷real B特徵
        feat_q_pool, _ = self.netF1(feat_q, self.opt.num_patches, sample_ids) #藉由判別器判斷fake A特徵
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals
