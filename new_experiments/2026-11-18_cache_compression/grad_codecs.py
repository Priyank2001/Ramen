"""
Gradient codecs: how a cache entry's value (its per-sample gradient, D numbers) is stored and read back.

Block-wise quantisation: the D values are split into blocks of `block` consecutive numbers (64 by default; every
LayerNorm weight / bias tensor of ViT-B has 768 = 12 x 64 entries, so no block straddles two parameter tensors) and
each block gets its own fp32 scale. Per-block scales keep small-gradient layers from being rounded to zero by a
large-gradient layer elsewhere in the vector.

    codec = Int8Codec()                      # or IdentityCodec(), Int4Codec(), SignCodec()
    store = codec.allocate(capacity, D, device)
    codec.write(store, slot, g)              # g: (D,) fp16
    g_hat = codec.read(store, idx)           # idx: LongTensor of any shape -> (*idx.shape, D) fp16
    codec.bytes_per_value(D)                 # storage per entry's value, in bytes

Codecs: IdentityCodec (fp16, Ramen's own storage), Int8Codec / Int4Codec (symmetric, round to nearest, levels
-127..127 / -7..7, scale = block abs-max / levels), SignCodec (1 bit: sign, read back as sign * block mean |g|).
"""
import torch

SCALE_DTYPE = torch.float32


class GradientCodec:
    name = None

    def __init__(self, block=64):
        self.block = block

    def allocate(self, capacity, D, device):
        raise NotImplementedError

    def write(self, store, slot, g):
        raise NotImplementedError

    def read(self, store, idx):
        raise NotImplementedError

    def bytes_per_value(self, D):
        raise NotImplementedError

    # helpers -----------------------------------------------------------------
    def _blocks(self, g):
        D = g.shape[-1]
        if D % self.block:
            raise ValueError(f'gradient dim {D} is not a multiple of block {self.block}')
        return g.float().reshape(*g.shape[:-1], D // self.block, self.block)

    def _scale_bytes(self, D):
        return (D // self.block) * SCALE_DTYPE.itemsize


class IdentityCodec(GradientCodec):
    """Ramen's storage: the fp16 gradient itself (reference; bitwise identical to PriorityCache.values)."""
    name = 'fp16'

    def allocate(self, capacity, D, device):
        return {'values': torch.empty((capacity, D), device=device, dtype=torch.half)}

    def write(self, store, slot, g):
        store['values'][slot] = g

    def read(self, store, idx):
        return store['values'][idx]

    def bytes_per_value(self, D):
        return D * 2


class _LevelCodec(GradientCodec):
    """Symmetric integer quantisation with `levels` steps each side of zero, one scale per block."""
    levels = None

    def quantise(self, g):
        b = self._blocks(g)
        scale = b.abs().amax(-1) / self.levels                        # (..., nb)
        q = torch.where(scale[..., None] > 0, b / scale[..., None].clamp_min(1e-38), torch.zeros_like(b))
        return q.round().clamp(-self.levels, self.levels).to(torch.int8), scale

    def dequantise(self, q, scale):
        g = q.float() * scale[..., None].float()
        return g.reshape(*g.shape[:-2], -1).half()


class Int8Codec(_LevelCodec):
    name = 'int8'
    levels = 127

    def allocate(self, capacity, D, device):
        return {'q': torch.zeros((capacity, D), device=device, dtype=torch.int8),
                'scale': torch.zeros((capacity, D // self.block), device=device, dtype=SCALE_DTYPE)}

    def write(self, store, slot, g):
        q, scale = self.quantise(g.reshape(-1))
        store['q'][slot] = q.reshape(-1)
        store['scale'][slot] = scale

    def read(self, store, idx):
        q = store['q'][idx]
        return self.dequantise(q.reshape(*q.shape[:-1], -1, self.block), store['scale'][idx])

    def bytes_per_value(self, D):
        return D + self._scale_bytes(D)


class Int4Codec(_LevelCodec):
    """Levels -7..7 packed two per byte: low nibble = even index, high nibble = odd index, stored as q + 8."""
    name = 'int4'
    levels = 7

    def allocate(self, capacity, D, device):
        return {'packed': torch.zeros((capacity, D // 2), device=device, dtype=torch.uint8),
                'scale': torch.zeros((capacity, D // self.block), device=device, dtype=SCALE_DTYPE)}

    @staticmethod
    def pack(q):
        u = (q.to(torch.int16) + 8).to(torch.uint8).reshape(*q.shape[:-1], -1, 2)
        return u[..., 0] | (u[..., 1] << 4)

    @staticmethod
    def unpack(p):
        lo, hi = (p & 0x0F), (p >> 4)
        return torch.stack((lo, hi), -1).reshape(*p.shape[:-1], -1).to(torch.int16).sub(8).to(torch.int8)

    def write(self, store, slot, g):
        q, scale = self.quantise(g.reshape(-1))
        store['packed'][slot] = self.pack(q.reshape(-1))
        store['scale'][slot] = scale

    def read(self, store, idx):
        q = self.unpack(store['packed'][idx])
        return self.dequantise(q.reshape(*q.shape[:-1], -1, self.block), store['scale'][idx])

    def bytes_per_value(self, D):
        return D // 2 + self._scale_bytes(D)


class SignCodec(GradientCodec):
    """1 bit per value (g >= 0), 8 per byte (bit i of byte j = value 8j + i); read back as sign * block mean |g|."""
    name = 'sign'

    def __init__(self, block=64):
        super().__init__(block)
        self._weights = {}

    def _bit_weights(self, device):
        if device not in self._weights:
            self._weights[device] = (2 ** torch.arange(8, device=device)).to(torch.uint8)
        return self._weights[device]

    def allocate(self, capacity, D, device):
        return {'bits': torch.zeros((capacity, D // 8), device=device, dtype=torch.uint8),
                'scale': torch.zeros((capacity, D // self.block), device=device, dtype=SCALE_DTYPE)}

    def pack(self, positive):
        w = self._bit_weights(positive.device)
        return (positive.reshape(*positive.shape[:-1], -1, 8).to(torch.uint8) * w).sum(-1, dtype=torch.uint8)

    def unpack(self, bits):
        w = self._bit_weights(bits.device)
        return ((bits[..., None] & w) > 0).reshape(*bits.shape[:-1], -1)

    def write(self, store, slot, g):
        g = g.reshape(-1)
        store['bits'][slot] = self.pack(g >= 0)
        store['scale'][slot] = self._blocks(g).abs().mean(-1)

    def read(self, store, idx):
        sign = self.unpack(store['bits'][idx]).float().mul(2).sub(1)
        sign = sign.reshape(*sign.shape[:-1], -1, self.block)
        g = sign * store['scale'][idx][..., None]
        return g.reshape(*g.shape[:-2], -1).half()

    def bytes_per_value(self, D):
        return D // 8 + self._scale_bytes(D)


CODECS = {c.name: c for c in (IdentityCodec, Int8Codec, Int4Codec, SignCodec)}
