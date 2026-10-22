import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from .optimizer import get_optimizer_class
from .clip_utils import encode_text, DEFAULT_TEMPLATES


class ModelForTTA:

    def __call__(self, x):
        return self.classify(self.featurize(x))

    def featurize(self, x):
        raise NotImplementedError

    def classify(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def reset_parameters(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def step_and_zero_grad(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


class Normalize(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim=self.dim)


class ToHalf(nn.Module):
    def forward(self, x):
        return x.half()


class CLIPModelForTTA(ModelForTTA):

    def __init__(self, model, class_names, cfg, args):
        super().__init__()

        # first generate text embeddings
        with torch.no_grad():
            templates = cfg.get('templates') or DEFAULT_TEMPLATES
            self.text_feature = encode_text(model, class_names, templates)

        num_classes, feat_dim = self.text_feature.shape
        linear = nn.Linear(feat_dim, num_classes, bias=False)
        linear.weight.data.copy_(self.text_feature * 100)
        linear = linear.to(device=args.device).half()

        self.model = nn.Sequential(
            ToHalf(),
            model.visual,
            Normalize(),
            linear,
        )

        self.device = model.visual.conv1.weight.device
        self.dtype = model.visual.conv1.weight.dtype

        # Only update normalization params
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)

                if args.use_tbn:
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

        self.feat_dim = self.text_feature.shape[-1]
        self.grad_dim = sum(param.shape[-1] for param in self.model.parameters() if param.requires_grad)

        print(self.grad_dim)

        # WARNING: The optimizer has to be a state-free one
        self.optimizer = get_optimizer_class(cfg['optimizer'])(self.model.parameters(), lr=cfg['lr'])

        # Initial state
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def featurize(self, x):
        return F.normalize(self.model[1](x.type(self.dtype)), dim=1)

    def classify(self, x):
        return 100.0 * x @ self.text_feature.T
