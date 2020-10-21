from models.networks.sync_batchnorm import DataParallelWithCallback
from models.indoor_model import IndoorModel


class PoseTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.indoor_model = IndoorModel(opt)
        if len(opt.gpu_ids) > 0:
            self.indoor_model = DataParallelWithCallback(self.indoor_model,
                                                         device_ids=opt.gpu_ids)
            self.indoor_model_on_one_gpu = self.indoor_model.module
        else:
            self.indoor_model_on_one_gpu = self.indoor_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.indoor_model_on_one_gpu.create_optimizers(opt)
            self.optimizer_P, self.optimizer_PL = self.indoor_model_on_one_gpu.create_pose_optimizer(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.indoor_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.indoor_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def run_posenet_one_step(self, data):
        self.optimizer_P.zero_grad()
        self.optimizer_PL.zero_grad()

        p_losses, pose = self.indoor_model(data, mode='train_pose')
        # p_loss = p_losses['pose'].mean()
        p_loss = sum(p_losses.values()).mean()
        p_loss.backward()
        self.optimizer_P.step()
        # self.optimizer_PL.step()
        self.p_losses = p_losses
        self.pose = pose

    def get_latest_losses(self, pose_phase):
        return {**self.p_losses} if pose_phase else {**self.g_losses, **self.d_losses}

    def get_latest_generated(self, pose_phase):
        return self.pose if pose_phase else self.generated

    def save(self, epoch, pose_phase):
        self.indoor_model_on_one_gpu.save(epoch, pose_phase)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch, pose_phase):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            if pose_phase:
                for param_group in self.optimizer_P.param_groups:
                    param_group['lr'] = new_lr_D
            else:
                for param_group in self.optimizer_D.param_groups:
                    param_group['lr'] = new_lr_D
                for param_group in self.optimizer_G.param_groups:
                    param_group['lr'] = new_lr_G

            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
