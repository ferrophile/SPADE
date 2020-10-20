from models.networks.sync_batchnorm import DataParallelWithCallback
from models.indoor_model import IndoorModel


class PoseTrainer():
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
            self.optimizer_P, self.optimizer_PL = self.indoor_model_on_one_gpu.create_pose_optimizer(opt)
            self.old_lr = opt.lr

    def train_one_step(self, data):
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

    def get_latest_losses(self):
        return {**self.p_losses}

    def get_latest_generated(self):
        return self.pose

    def save(self, epoch, pose_phase):
        self.indoor_model_on_one_gpu.save(epoch, pose_phase)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_P = new_lr
            else:
                new_lr_P = new_lr * 2

            for param_group in self.optimizer_P.param_groups:
                param_group['lr'] = new_lr_P
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
