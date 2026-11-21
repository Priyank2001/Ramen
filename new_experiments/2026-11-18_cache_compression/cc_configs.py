"""
Datasets and variants of the cache-compression experiment (plain data + one pure function).

Pre-registered 2026-11-18 before any run:
  * a codec is LOSSLESS-ENOUGH if |accuracy - Ramen| <= noise floor (domain-averaged, seed 0);
  * it BEATS SHRINKING if its accuracy exceeds its matched-memory control (`cc_cap_<codec>`: fp16 Ramen whose queue
    capacity is cut to the same cache bytes) by more than the noise floor.
  Compression is worth using only if it is lossless-enough and beats shrinking.

    from cc_configs import variants
    variants('ImageNetC5K')   # name -> ('RamenCompressed', cfg overrides on cfg/<dataset>/Ramen.yaml)
"""
from compressed_cache import bytes_per_entry
from grad_codecs import CODECS

# model / num_workers of the cached NoAdapt and Ramen runs (num_workers changes stream order), floor, Ramen's K
DATASETS = {
    'ImageNetC5K': dict(model='clip_vitbase16', num_workers=16, floor=0.54, capacity=75),
    'DomainNet': dict(model='clip_vitbase32', num_workers=16, floor=0.11, capacity=300),
    'EuroSAT': dict(model='clip_vitbase16', num_workers=16, floor=1.51, capacity=7500),
}
FEAT_DIM, GRAD_DIM = 512, 39_936     # ViT-B/16 and ViT-B/32: 512-d embedding, 26 LayerNorms x (768 weight + 768 bias) = 52 tensors of 768
BLOCK = 64
CODEC_LEVELS = ('int8', 'int4', 'sign')


def memory_ratio(codec_name):
    """Bytes per cache entry with this codec / with Ramen's fp16 storage."""
    fp16 = bytes_per_entry(CODECS['fp16'](BLOCK), FEAT_DIM, GRAD_DIM)
    return bytes_per_entry(CODECS[codec_name](BLOCK), FEAT_DIM, GRAD_DIM) / fp16


def matched_capacity(dataset, codec_name):
    """Queue capacity whose fp16 cache takes the same bytes as Ramen's capacity under this codec."""
    return max(1, round(DATASETS[dataset]['capacity'] * memory_ratio(codec_name)))


def variants(dataset):
    v = {'cc_off': ('RamenCompressed', {'codec': 'fp16', 'block': BLOCK})}   # must equal Ramen exactly
    for c in CODEC_LEVELS:
        v[f'cc_{c}'] = ('RamenCompressed', {'codec': c, 'block': BLOCK})
    for c in CODEC_LEVELS:
        v[f'cc_cap_{c}'] = ('RamenCompressed', {'codec': 'fp16', 'block': BLOCK,
                                                'max_capacity': matched_capacity(dataset, c)})
    return v
