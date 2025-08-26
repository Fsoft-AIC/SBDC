from argparse import Namespace, ArgumentParser

import torch
from torch import nn
import numpy as np
import scipy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from . import encoder


class BaseSSL(nn.Module):
    """
    Inspired by the PYTORCH LIGHTNING https://pytorch-lightning.readthedocs.io/en/latest/
    Similar but lighter and customized version.
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def get_ckpt(self):
        return {
            'state_dict': self.state_dict(),
            'hparams': self.hparams,
        }

    @classmethod
    def load(cls, ckpt, device=None):
        parser = ArgumentParser()
        cls.add_model_hparams(parser)
        hparams = parser.parse_args([], namespace=ckpt['hparams'])

        res = cls(hparams, device=device)
        res.load_state_dict(ckpt['state_dict'])
        return res

    @classmethod
    def default(cls, device=None, **kwargs):
        parser = ArgumentParser()
        cls.add_model_hparams(parser)
        hparams = parser.parse_args([], namespace=Namespace(**kwargs))
        res = cls(hparams, device=device)
        return res

    def forward(self, x):
        pass

    @staticmethod
    def add_parent_hparams(add_model_hparams):
        def foo(cls, parser):
            for base in cls.__bases__:
                base.add_model_hparams(parser)
            add_model_hparams(cls, parser)
        return foo

    @classmethod
    def add_model_hparams(cls, parser):
        parser.add_argument('--pretrain_on', help='Dataset to use', default='imagenet')


class SimCLR(BaseSSL):
    @classmethod
    @BaseSSL.add_parent_hparams
    def add_model_hparams(cls, parser):
        # loss params
        parser.add_argument('--temperature', default=0.1, type=float, help='Temperature in the NTXent loss')
        # data params
        parser.add_argument('--multiplier', default=2, type=int)
        parser.add_argument('--color_dist_s', default=1., type=float, help='Color distortion strength')
        # ddp
        parser.add_argument('--sync_bn', default=True, type=bool,
            help='Syncronises BatchNorm layers between all processes if True'
        )

    def __init__(self, hparams, device=None):
        super().__init__(hparams)

        self.hparams.dist = getattr(self.hparams, 'dist', 'dp')

        model = encoder.EncodeProject(hparams)
        self.reset_parameters()
        if device is not None:
            model = model.to(device)
        if self.hparams.dist == 'ddp':
            if self.hparams.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            dist.barrier()
            if device is not None:
                model = model.to(device)
            self.model = DDP(model, [hparams.gpu], find_unused_parameters=True)
        elif self.hparams.dist == 'dp':
            self.model = model
        else:
            raise NotImplementedError

    def reset_parameters(self):
        def conv2d_weight_truncated_normal_init(p):
            fan_in = p.shape[1]
            stddev = np.sqrt(1. / fan_in) / .87962566103423978
            r = scipy.stats.truncnorm.rvs(-2, 2, loc=0, scale=1., size=p.shape)
            r = stddev * r
            with torch.no_grad():
                p.copy_(torch.FloatTensor(r))

        def linear_normal_init(p):
            with torch.no_grad():
                p.normal_(std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_weight_truncated_normal_init(m.weight)
            elif isinstance(m, nn.Linear):
                linear_normal_init(m.weight)

    def encode(self, x):
        return self.model(x, out='h')

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_ckpt(self):
        return {
            'state_dict': self.model.module.state_dict(),
            'hparams': self.hparams,
        }

    def load_state_dict(self, state):
        k = next(iter(state.keys()))
        if k.startswith('model.module'):
            super().load_state_dict(state)
        else:
            self.model.load_state_dict(state)


class SSLEval(BaseSSL):
    @classmethod
    @BaseSSL.add_parent_hparams
    def add_model_hparams(cls, parser):
        parser.add_argument('--encoder_ckpt', default='', type=str, help='Path to the encoder checkpoint')
        parser.add_argument('--model_type', default='resnet50', type=str, help='Architecture of  encoder')
        parser.add_argument('--finetune', default=False, type=bool, help='Finetunes the encoder if True')

    def __init__(self, hparams, device=None, n_classes=10):
        super().__init__(hparams)

        self.hparams.dist = getattr(self.hparams, 'dist', 'dp')

        if hparams.encoder_ckpt != '':
            ckpt = torch.load(hparams.encoder_ckpt, map_location=device)
            if getattr(ckpt['hparams'], 'dist', 'dp') == 'ddp':
                ckpt['hparams'].dist = 'dp'
            if self.hparams.dist == 'ddp':
                ckpt['hparams'].dist = 'gpu:%d' % hparams.gpu
            ckpt['hparams'].arch = self.hparams.model_type
            self.encoder = REGISTERED_MODELS[ckpt['hparams'].problem].load(ckpt, device=device)
        else:
            raise 'Random encoder is used!!! Provide a pretrained SimCLR ckpt to --encoder_ckpt'
            # self.encoder = SimCLR.default(device=device)

        self.encoder.model.to(device)

        if not hparams.finetune:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif hparams.dist == 'ddp':
            raise NotImplementedError

        self.encoder.eval()
        if hparams.pretrain_on == 'cifar':
            hdim = self.encode(torch.ones(10, 3, 32, 32).to(device)).shape[1]
        elif hparams.pretrain_on == 'imagenet':
            hdim = self.encode(torch.ones(10, 3, 224, 224).to(device)).shape[1]

        model = nn.Linear(hdim, n_classes).to(device)
        model.weight.data.zero_()
        model.bias.data.zero_()
        self.model = model

        if hparams.dist == 'ddp':
            self.model = DDP(model, [hparams.gpu])

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x, out='h')

    def forward(self, x):
        h = self.encode(x)
        return self.model(h)

    def get_ckpt(self):
        return {
            'state_dict': self.state_dict() if self.hparams.finetune else self.model.state_dict(),
            'hparams': self.hparams,
        }

    def load_state_dict(self, state):
        if self.hparams.finetune:
            super().load_state_dict(state)
        else:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(state)
            else:
                self.model.load_state_dict(state)


def configure_optimizers(args, model, cur_iter=-1):
    iters = args.iters

    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if args.opt == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]

    LR = args.lr

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
        )
    else:
        raise NotImplementedError

    scheduler = None

    # if args.verbose:
    #     print('Optimizer : ', optimizer)
    #     print('Scheduler : ', scheduler)

    return optimizer, scheduler


REGISTERED_MODELS = {
    'sim-clr': SimCLR,
    'eval': SSLEval,
}