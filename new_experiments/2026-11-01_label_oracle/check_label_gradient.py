"""
Check that the label value points the right way: one signSGD step on each image's OWN stored value should raise
the probability of the class that value is meant to sharpen (the true label for the CE value, the zero-shot class
for Ramen's entropy value). One batch of 100 DomainNet images (seed 0, smoke subset), no cache involved.

    CUDA_VISIBLE_DEVICES=2 python check_label_gradient.py | tee logs/check_label_gradient.log
"""
import torch
from torch.utils.data import ConcatDataset, DataLoader

from label_oracle import VARIANTS, LabelOracleExperiment


def main():
    exp = LabelOracleExperiment('DomainNet', seed=0, n_per_domain=200)
    exp.exp.load()
    loader = DataLoader(ConcatDataset(list(exp.exp.datasets)), batch_size=100, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    x, y = next(iter(loader))[:2]
    method = exp.exp.build_method('RamenLabelValues', VARIANTS['labeloracle_true_ce'][1])
    model = method.model
    x, y = x.to(method.device), y.to(method.device)

    logits = model.classify(model.featurize(x))
    zs = logits.argmax(-1)
    method.loss_fn(logits).backward(retain_graph=True)
    values = {'entropy (Ramen)': (model.get_by_sample_grad(), zs)}
    values['CE toward true label (oracle)'] = (method.value_rule.revealed(model, logits, y), y)
    cos = torch.nn.functional.cosine_similarity(*(v[0].float() for v in values.values())).mean().item()
    p0 = logits.float().softmax(1)

    print(f'{len(y)} images; zero-shot correct on {(zs == y).sum().item()}; '
          f'mean cosine(entropy value, CE value) = {cos:+.3f}')
    for name, (g, target) in values.items():
        method.loss_fn(model.classify(model.featurize(x))).backward()  # .grad exists again (zero_grad sets None)
        model.set_by_sample_grad(g)
        model.step_and_zero_grad()
        with torch.no_grad():
            p1 = model(x).float().softmax(1)
        model.reset_parameters()
        d = (p1.gather(1, target[:, None]) - p0.gather(1, target[:, None])).squeeze(1)
        print(f'{name}: p(class it targets) rises on {(d > 0).sum().item()} of {len(y)}, mean {d.mean().item():+.3f}; '
              f'correct {(p0.argmax(1) == y).sum().item()} -> {(p1.argmax(1) == y).sum().item()} of {len(y)}')


if __name__ == '__main__':
    main()
