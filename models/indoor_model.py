from models.pix2pix_model import Pix2PixModel
import torch
import models.networks as networks
import util.util as util
import util.transformation as trans_utils


class IndoorModel(Pix2PixModel):
    def __init__(self, opt):
        torch.nn.Module.__init__(self)
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.use_network = {}
        self.netG, self.netD, self.netE, self.netT = self.initialize_networks(opt)
        # self.criterionBoxes = torch.nn.L1Loss()
        self.criterionBoxes = networks.ModifiedL1Loss()

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    def initialize_networks(self, opt):
        use_network = {'G': True, 'D': True, 'E': True, 'T': True}
        if opt.isTrain and opt.train_phase == 'transform':
            for net in ['G', 'D', 'E']:
                use_network[net] = False
        if not opt.isTrain and opt.transform_inference:
            for net in ['G', 'D', 'E']:
                use_network[net] = False
        if not opt.isTrain:
            use_network['D'] = False
        if not opt.use_transform:
            use_network['T'] = False
        self.use_network = use_network

        load_network = use_network.copy()
        if opt.isTrain and not opt.continue_train:
            if opt.train_phase == 'transform':
                load_network['T'] = False
            else:
                for net in ['G', 'D', 'E']:
                    load_network[net] = False

        netG = networks.define_G(opt) if use_network['G'] else None
        netD = networks.define_D(opt) if use_network['D'] else None
        netE = networks.define_E(opt) if use_network['E'] else None
        netT = networks.define_T(opt) if use_network['T'] else None

        if load_network['G']:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
        if load_network['D']:
            netD = util.load_network(netD, 'D', opt.which_epoch, opt)
        if load_network['E']:
            netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        if load_network['T']:
            netT = util.load_network(netT, 'T', opt.transform_epoch, opt)

        return netG, netD, netE, netT

    def preprocess_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
            for k in data.keys():
                if k != 'path':
                    data[k] = data[k].cuda()

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

        return input_semantics, data['image'], data['empty_image'], data['instance'], data['semantic']

    def forward(self, data, mode):
        input_semantics, real_image, empty_image, instances, fine_semantics = self.preprocess_input(data)

        generated = {}
        if mode == 'generator':
            if self.opt.use_transform:
                with torch.no_grad():
                    _, generated['input'], generated['out_sem'] = self.generate_transform(
                        empty_image, instances, fine_semantics, False)
                g_loss, generated['out_result'] = self.compute_generator_loss(generated['out_sem'], real_image, empty_image)
            else:
                generated['input'] = input_semantics
                g_loss, generated['out_result'] = self.compute_generator_loss(input_semantics, real_image, empty_image)

            return g_loss, generated
        elif mode == 'discriminator':
            if self.opt.use_transform:
                with torch.no_grad():
                    _, _, out_sem = self.generate_transform(empty_image, instances, fine_semantics, False)
                d_loss = self.compute_discriminator_loss(out_sem, real_image, empty_image)
            else:
                d_loss = self.compute_discriminator_loss(input_semantics, real_image, empty_image)

            return d_loss
        elif mode == 'transform_generator':
            generated = {}
            tg_loss, generated['input'], generated['out_sem'] = self.compute_transform_generator_loss(
                empty_image, instances, fine_semantics)
            return tg_loss, generated
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(empty_image)
            return mu, logvar
        elif mode == 'transform_inference':
            with torch.no_grad():
                _, generated['input'], generated['out_sem'] = self.generate_transform(
                    empty_image, instances, fine_semantics, True)
            return generated
        elif mode == 'inference':
            with torch.no_grad():
                generated['input'] = input_semantics
                generated['out_result'], _ = self.generate_fake(input_semantics, empty_image)
            return generated
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        if opt.train_phase == 'generate':
            return Pix2PixModel.create_optimizers(self, opt)

        T_params = list(self.netT.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            T_lr = opt.lr
        else:
            T_lr = opt.lr / 2

        optimizer_T = torch.optim.Adam(T_params, lr=T_lr, betas=(beta1, beta2))
        return optimizer_T, None

    def save(self, epoch):
        if self.use_network['G']:
            util.save_network(self.netG, 'G', epoch, self.opt)
        if self.use_network['D']:
            util.save_network(self.netD, 'D', epoch, self.opt)
        if self.use_network['E']:
            util.save_network(self.netE, 'E', epoch, self.opt)
        if self.use_network['T']:
            util.save_network(self.netT, 'T', epoch, self.opt)

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

    def compute_transform_generator_loss(self, empty_image, instances, fine_semantics):
        TG_losses = {}

        loss_tensors, input_tensors, output_tensors = self.generate_transform(
            empty_image, instances, fine_semantics, True
        )

        TG_losses['Boxes'] = self.criterionBoxes(loss_tensors)
        return TG_losses, input_tensors, output_tensors

    def generate_transform(self, empty_image, instances, fine_semantics, filter_classes):
        fine_tensors, box_tensors, semantic_classes = \
            trans_utils.instances_to_boxes(instances, fine_semantics, filter_classes)
        fine_tensor = torch.cat(fine_tensors, dim=0)

        box_counts = [len(b) for b in box_tensors]
        box_tensor = torch.cat(box_tensors, dim=0)
        pred_box_tensor = self.netT(box_tensor, empty_image, box_counts)
        pred_box_tensors = torch.split(pred_box_tensor, box_counts)

        loss_tensors = {
            'pred_box': pred_box_tensor,
            'box': box_tensor,
            'fine': fine_tensor
        }

        input_tensors = [trans_utils.boxes_to_labels(pb, s) for pb, s in zip(box_tensors, semantic_classes)]
        input_tensors = torch.stack(input_tensors, dim=0)

        output_tensors = [trans_utils.boxes_to_labels(pb, s) for pb, s in zip(pred_box_tensors, semantic_classes)]
        output_tensors = torch.stack(output_tensors, dim=0)

        return loss_tensors, input_tensors, output_tensors

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
