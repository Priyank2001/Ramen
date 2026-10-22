import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimizer import get_optimizer_class
from .clip_utils import encode_text, DEFAULT_TEMPLATES


class BySampleLayerNorm(nn.Module):
    """
    Reparameterization trick for computing a batch of gradient at a time
    """

    def __init__(self, layernorm_module, max_batch_size=100, batch_first=True):
        super().__init__()

        self.layernorm_module = layernorm_module
        self.max_batch_size = max_batch_size
        self.batch_first = batch_first

        # Save pretrained affine parameters in buffer, they won't be updated
        self.register_buffer('weight', layernorm_module.weight.data)
        self.register_buffer('bias', layernorm_module.bias.data)

        # Weight and bias by sample ...
        self.weight_by_sample = nn.Parameter(self.weight.expand(max_batch_size, -1).contiguous())
        self.bias_by_sample = nn.Parameter(self.bias.expand(max_batch_size, -1).contiguous())
        self.recent_B = None

        # skip affine transform, use our implementation instead
        self.layernorm_module.weight = None
        self.layernorm_module.bias = None

    def forward(self, x):

        if self.batch_first:
            B = x.shape[0]  # B * L * D
        else:
            B = x.shape[-2]  # L * B * D

        assert B <= self.max_batch_size, f'batch size {B} must be smaller than max batch size {self.max_batch_size}'

        self.recent_B = B  # save for gradient operations

        orig_type = x.dtype
        x = x.type(torch.float32)

        x_norm = self.layernorm_module(x)

        if self.batch_first:
            view_shape = [B] + [1] * (len(x_norm.shape) - 2) + [-1]
        else:
            view_shape = [1] * (len(x_norm.shape) - 2) + [B, -1]

        ret = x_norm * self.weight_by_sample[:B].view(*view_shape) + self.bias_by_sample[:B].view(*view_shape)

        return ret.type(orig_type)

    def reset_parameters(self):
        """
        Reset to pre-trained initialization
        :return:
        """
        self.weight_by_sample.copy_(self.weight.expand(self.max_batch_size, -1))
        self.bias_by_sample.copy_(self.bias.expand(self.max_batch_size, -1))
        self.recent_B = None


class BySampleBatchNorm(nn.Module):
    """
    Reparameterization trick for computing a batch of gradient at a time
    """

    def __init__(self, batchnorm_module, max_batch_size=100):
        super().__init__()

        self.batchnorm_module = batchnorm_module
        self.max_batch_size = max_batch_size

        # Save pretrained affine parameters in buffer, they won't be updated
        self.register_buffer('weight', batchnorm_module.weight.data)
        self.register_buffer('bias', batchnorm_module.bias.data)

        # Weight and bias by sample ...
        self.weight_by_sample = nn.Parameter(self.weight.expand(max_batch_size, -1).contiguous())
        self.bias_by_sample = nn.Parameter(self.bias.expand(max_batch_size, -1).contiguous())
        self.recent_B = None

        # skip affine transform, use our implementation instead
        self.batchnorm_module.weight = None
        self.batchnorm_module.bias = None

    def forward(self, x):
        B = x.shape[0]

        assert B <= self.max_batch_size, f'batch size {B} must be smaller than max batch size {self.max_batch_size}'

        self.recent_B = B  # save for gradient operations

        orig_type = x.dtype
        x = x.type(torch.float32)

        x_norm = self.batchnorm_module(x)

        view_shape = [B, -1, 1, 1]

        ret = x_norm * self.weight_by_sample[:B].view(*view_shape) + self.bias_by_sample[:B].view(*view_shape)

        return ret.type(orig_type)

    def reset_parameters(self):
        """
        Reset to pre-trained initialization
        :return:
        """
        self.weight_by_sample.copy_(self.weight.expand(self.max_batch_size, -1))
        self.bias_by_sample.copy_(self.bias.expand(self.max_batch_size, -1))
        self.recent_B = None


def replace_norm_with_custom(module, orig_cls, custom_cls, **kwargs):
    """
    Replace normalization layers in module with a customized layer
    :param module:
    :param custom_cls:
    :return:
    """

    for name, child in module.named_children():
        if isinstance(child, orig_cls) and hasattr(child, 'weight') and child.weight is not None:
            new_ln = custom_cls(child, **kwargs)
            setattr(module, name, new_ln)
        else:
            replace_norm_with_custom(child, orig_cls, custom_cls, **kwargs)


class ModelForBySampleTTA:

    def __call__(self, x):
        return self.classify(self.featurize(x))

    def featurize(self, x):
        raise NotImplementedError

    def classify(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def get_by_sample_grad(self):
        """
        Get sample-wise gradient
        :return:
        """
        grad_matrix = []

        for module in self.model.modules():
            if isinstance(module, (BySampleLayerNorm, BySampleBatchNorm)):
                grad_matrix.append(module.weight_by_sample.grad[:module.recent_B])
                grad_matrix.append(module.bias_by_sample.grad[:module.recent_B])

        grad_matrix = torch.cat(grad_matrix, dim=1)

        return grad_matrix

    @torch.no_grad()
    def set_by_sample_grad(self, grad_matrix):

        idx = 0

        for module in self.model.modules():
            if isinstance(module, (BySampleLayerNorm, BySampleBatchNorm)):
                module.weight_by_sample.grad[:module.recent_B].copy_(
                    grad_matrix[:module.recent_B, idx: idx + module.weight_by_sample.shape[1]])
                idx += module.weight_by_sample.shape[1]

                module.bias_by_sample.grad[:module.recent_B].copy_(
                    grad_matrix[:module.recent_B, idx: idx + module.bias_by_sample.shape[1]])
                idx += module.bias_by_sample.shape[1]

        assert idx == grad_matrix.shape[1]  # make sure all items are used

    @torch.no_grad()
    def reset_parameters(self):
        for module in self.model.modules():
            if isinstance(module, (BySampleLayerNorm, BySampleBatchNorm)):
                module.reset_parameters()

    def step_and_zero_grad(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


class CLIPModelForBySampleTTA(ModelForBySampleTTA):

    def __init__(self, model, class_names, cfg, args):
        super().__init__()

        # first generate text embeddings
        with torch.no_grad():
            templates = cfg.get('templates') or DEFAULT_TEMPLATES
            self.text_feature = encode_text(model, class_names, templates)

        self.model = model.visual
        self.model_type = 'vit' if hasattr(self.model, 'transformer') else 'rn'
        self.device = model.visual.conv1.weight.device
        self.dtype = model.visual.conv1.weight.dtype

        if self.model_type == 'vit':  # ViT model
            # LayerNorms within visual.transformer take L * B * D as input
            replace_norm_with_custom(self.model.transformer, nn.LayerNorm, BySampleLayerNorm,
                                     max_batch_size=args.max_batch_size, batch_first=False)

            # LayerNorms out of visual.transformer (ln_pre and ln_post) is takes B * L * D as input
            replace_norm_with_custom(self.model, nn.LayerNorm, BySampleLayerNorm,
                                     max_batch_size=args.max_batch_size, batch_first=True)

        else:  # RN model
            replace_norm_with_custom(self.model, nn.BatchNorm2d, BySampleBatchNorm,
                                     max_batch_size=args.max_batch_size)

        # Only update normalization affine params
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, (BySampleLayerNorm, BySampleBatchNorm)):
                m.weight_by_sample.requires_grad_(True)
                m.bias_by_sample.requires_grad_(True)

        self.feat_dim = self.text_feature.shape[-1]
        self.grad_dim = sum(param.shape[-1] for param in self.model.parameters() if param.requires_grad)

        # WARNING: The optimizer has to be a state-free one
        self.optimizer = get_optimizer_class(cfg['optimizer'])(self.model.parameters(), lr=cfg['lr'])

    def featurize(self, x):
        return F.normalize(self.model(x.type(self.dtype)), dim=1)

    def classify(self, x):
        return 100.0 * x @ self.text_feature.T
