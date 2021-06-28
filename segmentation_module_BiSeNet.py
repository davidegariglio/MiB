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
    model = IncrementalSegmentationBiSeNet(classes=classes)
    return model

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationBiSeNet(nn.Module):
    def __init__(self, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationBiSeNet, self).__init__()
        self.core = BiSeNet('resnet50')
        channels_out = 128
        channels_1 = 1024
        channels_2 = 2048
        assert isinstance(classes, list), "Classes must be a list where to every index correspond the num of classes for that task"
        self.cls = nn.ModuleList([nn.Conv2d(channels_out, c, 1) for c in classes])
        self.sv1 = nn.ModuleList([nn.Conv2d(channels_1, c, 1) for c in classes])
        self.sv2 = nn.ModuleList([nn.Conv2d(channels_2, c, 1) for c in classes])

        self.classes = classes
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False):

        features_ffm, features_1, features_2 = self.core(x)
        self.features = features_ffm  # save features for dist. loss
        out = []
        out_1 = []
        out_2 = []
        for mod in self.cls:
            out.append(mod(features_ffm))
        x_o = torch.cat(out, dim=1)

        if self.training == False:
            return x_o

        for mod in self.sv1:
            out_1.append(mod(features_1))
        out_1 = torch.cat(out_1, dim=1)

        for mod in self.sv2:
            out_2.append(mod(features_2))
        out_2 = torch.cat(out_2, dim=1)

        return x_o, out_1, out_2
    
    
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
        if self.training:
            out, out_sv1, out_sv2 = self._network(x)
            out = functional.interpolate(out, scale_factor=8, mode="bilinear")
            out_sv1 = functional.interpolate(out_sv1, size=out_size, mode="bilinear", align_corners=False)
            out_sv2 = functional.interpolate(out_sv2, size=out_size, mode="bilinear", align_corners=False)
            return out, out_sv1, out_sv2
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
                
    if __name__ == "__main__":
        model = make_model(classes=[2, 3])
        print(model)
        model.init_new_classifier("cpu")
