import torch
import torch.nn as nn
import torch.nn.functional as F


def get_similarity(key_embeds, ref_embeds, tau=-1, norm=False):
    pre_norm = True if tau > 0 else False
    if norm or pre_norm:
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
    sim = torch.mm(key_embeds, ref_embeds.t())
    if tau > 0:
        sim /= tau
    return sim