import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

# Prompt ensemble used unless the config sets `templates` (e.g. cfg/EuroSAT uses a satellite-photo prompt)
DEFAULT_TEMPLATES = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]


def encode_text(clip_model, class_names, templates, aggregate='average', return_text=False):
    all_texts = [[t.format(c.replace('_', ' ')) for t in templates] for c in class_names]
    text_features = []
    device = next(clip_model.parameters()).device
    for texts in all_texts:
        tokens = clip.tokenize(texts).to(device)
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb, dim=1)
        if aggregate == 'average':
            emb = emb.mean(dim=0)
        text_features.append(emb)
    text_features = torch.stack(text_features, dim=0)
    text_features = F.normalize(text_features, dim=-1)
    if return_text:
        return text_features, all_texts
    return text_features


def encode_text_single(clip_model, class_names, template):
    texts = [template.format(c.replace('_', ' ')) for c in class_names]
    device = next(clip_model.parameters()).device
    tokens = clip.tokenize(texts).to(device)
    emb = clip_model.encode_text(tokens)
    emb = F.normalize(emb, dim=1)
    return emb
