import torch
import torch.nn as nn
import models
from modules import build_contextpath
from modules.build_BiSeNet import  BiSeNet
from torch import distributed
import torch.nn.functional as functional

from functools import partial, reduce



from utils.logger import Logger

#this is more efficient
def make_model(opts=None, classes=None):
    model = IncrementalBiseNet(classes=classes)
    return model

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationBiSeNet(nn.Module):

    def __init__(self, body, head, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationBiSeNet, self).__init__()

        self.body = body ###
        self.head = head ###

        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"

        self.cls = nn.ModuleList(
            [nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1) for c in classes]
            # [nn.Conv2d(256, c, 1) for c in classes]
        )

        self.classes = classes
        self.head_channels = 256
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False):

        
        x_pl, xc1, xc2 = self.head(x)


        out = []

        for mod in self.cls:
            out.append(mod(x_pl))

        x_o = torch.cat(out, dim=1)

        if ret_intermediate:
            return x_o, x_b,  x_pl
        return x_o

    def init_new_classifier(self, device):
        cls = self.cls[-1]

        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False):
        out_size = x.shape[-2:]

        out = self._network(x, ret_intermediate)

        sem_logits = out[0] if ret_intermediate else out

        sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

        if ret_intermediate:
            return sem_logits, {"body": out[1], "pre_logits": out[2]}

        return sem_logits, {}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
