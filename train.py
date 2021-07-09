import torch
from tqdm import tqdm
# from torch import distributed
import torch.nn as nn
# from apex import amp
from torch.cuda import amp
from functools import reduce

from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss
from utils import get_regularizer
from torch.cuda.amp import autocast


class Trainer:
    def __init__(self, model, model_old, device, opts, trainer_state=None, classes=None):

        self.model_old = model_old
        self.model = model
        self.device = device
        self.batch_size = opts.batch_size
        self.scaler = amp.GradScaler()
        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
        else:
            self.old_classes = 0

        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance

        self.ret_intermediate = self.lde

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=90, logger=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
        
        tqlt = tqdm(total=len(train_loader)*self.batch_size)
        #tqlt = tqlt * batch_size
        device = self.device
        model = self.model
        criterion = self.criterion

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        scaler = amp.GradScaler()
        model.train()
        for cur_step, (images, labels) in enumerate(train_loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            with amp.autocast():
                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old = self.model_old(images)
                        features_old = self.model_old.features

                optim.zero_grad()
                with autocast():
                    outputs, out_1, out_2 = model(images)
                # features = model.features
                # xxx BCE / Cross Entropy Loss
                
                with autocast():
                    if not self.icarl_only_dist:
                        new_loss = criterion(outputs, labels)  # B x H x W

                    else:
                        new_loss = self.licarl(
                            outputs, labels, torch.sigmoid(outputs_old))
                     

                    loss = new_loss.mean()  # scalar

                    if self.icarl_combined:
                        # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                        n_cl_old = outputs_old.shape[1]
                        # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                        l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                      torch.sigmoid(outputs_old))
                    # distillation
                    if self.lde_flag:
                        lde = self.lde * self.lde_loss(features, features_old)

                    if self.lkd_flag:
                        # resize new output to remove new logits and keep only the old ones
                        lkd = self.lkd * self.lkd_loss(outputs, outputs_old)

                    # xxx first backprop of previous loss (compute the gradients for regularization methods)
                    loss_tot = loss + lkd + lde + l_icarl

            self.scaler.scale(loss_tot).backward()
            # with amp.scale_loss(loss_tot, optim) as scaled_loss:
            #     scaled_loss.backward()

            # xxx Regularizer (EWC, RW, PI) # What?

            if self.regularizer_flag:
                # if distributed.get_rank() == 0:
                self.regularizer.update()
                l_reg = self.reg_importance * self.regularizer.penalty()
                if l_reg != 0.:
                    # with amp.scale_loss(l_reg, optim) as scaled_loss:
                    self.scaler.scale(l_reg).backward()
                    
            self.scaler.step(optim)        
            self.scaler.update()
            if scheduler is not None:
                scheduler.step()
                
            # TODO: check this loss below:
            loss = new_loss.mean()
            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                            f" Loss={interval_loss}")
                logger.debug(
                    f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss', interval_loss, x)
                interval_loss = 0.0
            tqlt.update(len(labels))
        tqlt.close()

        # # collect statistics from multiple processes
        # epoch_loss = torch.tensor(epoch_loss).to(self.device)
        # reg_loss = torch.tensor(reg_loss).to(self.device)

        # torch.distributed.reduce(epoch_loss, dst=0)
        # torch.distributed.reduce(reg_loss, dst=0)

        # if distributed.get_rank() == 0:
        #     epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
        #     reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)
        
        epoch_loss = epoch_loss / len(train_loader)
        reg_loss = reg_loss / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")
        epoch_loss = epoch_loss / len(train_loader)
        reg_loss = reg_loss / len(train_loader)
        return (epoch_loss, reg_loss)

    def validate(self, loader, metrics, ret_samples_ids=None, logger=None):
        tqlt = tqdm(total=len(loader)*self.batch_size)
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old = self.model_old(images)
                        features_old = self.model_old.features

                # TODO: review return in validate inside the model
                outputs, dictionary = model(images)

                # features = model.features
                # xxx BCE / Cross Entropy Loss
                if not self.icarl_only_dist:
                    new_loss = criterion(outputs, labels)  # B x H x W
                    #loss_1 = criterion(out_1, labels)
                    #loss_2 = criterion(out_2, labels)
                else:
                    new_loss = self.licarl(
                        outputs, labels, torch.sigmoid(outputs_old))
                    #loss_1 = self.licarl(
                        #out_1, labels, torch.sigmoid(outputs_old))
                    #loss_2 = self.licarl(
                        #out_2, labels, torch.sigmoid(outputs_old))

                #loss = loss + loss_1 + loss_2
                loss = new_loss.mean()  # scalar

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                  torch.sigmoid(outputs_old))

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde_loss(features, features_old)

                if self.lkd_flag:
                    lkd = self.lkd_loss(outputs, outputs_old)

                # xxx Regularizer (EWC, RW, PI) # WHAT?
                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item()

                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0],
                                        prediction[0]))
                tqlt.update(len(labels))

            tqlt.close()

            # # collect statistics from multiple processes #Why
            # metrics.synch(device)
            score  = metrics.get_results()

            # class_loss = torch.tensor(class_loss).to(self.device)
            # reg_loss = torch.tensor(reg_loss).to(self.device)

            # torch.distributed.reduce(class_loss, dst=0)
            # torch.distributed.reduce(reg_loss, dst=0)

            # if distributed.get_rank() == 0:
            class_loss = torch.tensor(class_loss).cuda()
            reg_loss = torch.tensor(reg_loss).cuda()


            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)")

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])
