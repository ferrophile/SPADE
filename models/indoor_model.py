from models.pix2pix_model import Pix2PixModel
import torch
import models.networks as networks
import util.util as util


class IndoorModel(Pix2PixModel):
    def __init__(self, opt):
        torch.nn.Module.__init__(self)
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE, self.netP = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

            self.criterionPose = networks.PoseLoss()
            self.criterionPose = self.criterionPose.cuda()

    def initialize_networks(self, opt):
        netG, netD, netE = Pix2PixModel.initialize_networks(self, opt)
        netP = networks.define_P(opt)
        if not opt.isTrain or opt.continue_train:
            netP = util.load_network(netP, 'P', opt.which_epoch, opt)

        return netG, netD, netE, netP

    def create_pose_optimizer(self, opt):
        P_params = list(self.netP.parameters())
        P_loss_params = list(self.criterionPose.parameters())
        # P_all_params = P_params + P_loss_params

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            P_lr = opt.lr
        else:
            P_lr = opt.lr * 2

        optimizer_P = torch.optim.Adam(P_params, lr=P_lr, betas=(beta1, beta2))
        optimizer_PL = torch.optim.Adam(P_loss_params, lr=P_lr, betas=(beta1, beta2))
        return optimizer_P, optimizer_PL

    def save(self, epoch, pose_phase=False):
        if pose_phase:
            util.save_network(self.netP, 'P1', epoch, self.opt)
        else:
            util.save_network(self.netG, 'G', epoch, self.opt)
            util.save_network(self.netD, 'D', epoch, self.opt)
            if self.opt.use_vae:
                util.save_network(self.netE, 'E', epoch, self.opt)
            util.save_network(self.netP, 'P2', epoch, self.opt)

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label']
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['empty_image'] = data['empty_image'].cuda()
            data['pose'] = data['pose'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        
        if label_map.shape[1] == 1:
            label_map = label_map.long()
            input_semantics = input_label.scatter_(1, label_map, 1.0)
        else:
            input_semantics = label_map

        return input_semantics, data['image'], data['empty_image'], data['pose']

    def forward(self, data, mode):
        input_semantics, real_image, empty_image, pose = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(input_semantics, real_image, empty_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, real_image, empty_image)
            return d_loss
        elif mode == 'train_pose':
            p_loss, pred_pose = self.compute_perspective_loss(real_image, pose)
            return p_loss, pred_pose
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(empty_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, empty_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def compute_generator_loss(self, input_semantics, real_image, empty_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(input_semantics, empty_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, empty_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, empty_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def compute_perspective_loss(self, real_image, real_pose):
        losses = {}

        pred_pose = self.netP(real_image)
        q_pose = self.criterionPose(pred_pose, real_pose)
        # losses['t_pose'] = t_pose
        losses['q_pose'] = q_pose

        # losses['t_coeff'] = self.criterionPose.t_coeff.data
        # losses['q_coeff'] = self.criterionPose.q_coeff.data
        return losses, pred_pose

    def generate_fake(self, input_semantics, empty_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(empty_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        if self.opt.bg_con:
            fake_image = self.netG(input_semantics, z=z, bg=empty_image)
        else:
            fake_image = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss
