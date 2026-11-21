"""
Correctness tests for grad_codecs.py, compressed_cache.py and cc_configs.py (no model, no data; GPU if available,
since Ramen's cache is fp16 on the GPU).

    ~/miniconda3/envs/RAMEN/bin/python -m unittest -v test_compression
"""
import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src'))
sys.path.insert(0, HERE)

from cc_configs import BLOCK, DATASETS, FEAT_DIM, GRAD_DIM, matched_capacity, memory_ratio  # noqa: E402
from compressed_cache import CompressedPriorityCache, bytes_per_entry  # noqa: E402
from grad_codecs import IdentityCodec, Int4Codec, Int8Codec, SignCodec  # noqa: E402
from methods.Ramen import PriorityCache  # noqa: E402

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = 64 * 12      # a few blocks; production D = 39,936 = 624 blocks


def grads(n, D=D, seed=0):
    """fp16 gradients whose blocks have very different scales (as LayerNorm layers do)."""
    g = torch.Generator().manual_seed(seed)
    scale = 10 ** torch.empty(n, D // 64, 1).uniform_(-6, 0, generator=g)
    return (torch.randn(n, D // 64, 64, generator=g) * scale).reshape(n, D).half().to(DEV)


def roundtrip(codec, g):
    store = codec.allocate(len(g), g.shape[1], DEV)
    for i in range(len(g)):
        codec.write(store, i, g[i])
    return codec.read(store, torch.arange(len(g), device=DEV)), store


class TestCodecs(unittest.TestCase):
    def test_identity_is_bitwise(self):
        g = grads(5)
        out, _ = roundtrip(IdentityCodec(), g)
        self.assertTrue(torch.equal(out, g))

    def _level_error(self, codec, levels):
        g = grads(20)
        out, store = roundtrip(codec, g)
        b = g.float().reshape(20, -1, 64)
        scale = b.abs().amax(-1, keepdim=True) / levels
        err = (out.float().reshape(20, -1, 64) - b).abs()
        # rounding to the nearest level (<= scale / 2) plus fp16 rounding of the decoded value
        bound = scale / 2 + b.abs() * 2 ** -10 + 1e-7
        self.assertTrue(bool((err <= bound).all()), f'max excess {(err - bound).max().item():.3g}')
        # the block maximum is exact up to fp16 rounding, so no block is quantised to all zeros
        nonzero = (b.abs().amax(-1) > 0)
        self.assertTrue(bool((out.float().reshape(20, -1, 64).abs().amax(-1)[nonzero] > 0).all()))

    def test_int8_error_bound(self):
        self._level_error(Int8Codec(BLOCK), 127)

    def test_int4_error_bound(self):
        self._level_error(Int4Codec(BLOCK), 7)

    def test_int4_pack_unpack_exact(self):
        q = torch.arange(-7, 8, device=DEV, dtype=torch.int8).repeat(10)[:144].reshape(2, 72)
        self.assertTrue(torch.equal(Int4Codec.unpack(Int4Codec.pack(q)), q))

    def test_sign_pack_unpack_exact(self):
        c = SignCodec(BLOCK)
        pos = torch.rand(3, 128, device=DEV) > 0.5
        self.assertTrue(torch.equal(c.unpack(c.pack(pos)), pos))

    def test_sign_decodes_to_sign_times_block_mean(self):
        g = grads(10)
        out, _ = roundtrip(SignCodec(BLOCK), g)
        b, o = g.float().reshape(10, -1, 64), out.float().reshape(10, -1, 64)
        self.assertTrue(torch.equal(o >= 0, b >= 0))
        ref = b.abs().mean(-1, keepdim=True).expand_as(o)
        torch.testing.assert_close(o.abs(), ref.half().float(), rtol=0, atol=0)

    def test_zero_block_stays_zero(self):
        g = grads(2)
        g[0, :64] = 0
        for codec in (Int8Codec(BLOCK), Int4Codec(BLOCK)):
            out, _ = roundtrip(codec, g)
            self.assertTrue(bool((out[0, :64] == 0).all()), codec.name)

    def test_read_any_index_shape(self):
        g = grads(6)
        for codec in (IdentityCodec(), Int8Codec(BLOCK), Int4Codec(BLOCK), SignCodec(BLOCK)):
            full, store = roundtrip(codec, g)
            idx = torch.tensor([[5, 0, 2], [1, 1, 4]], device=DEV)
            self.assertEqual(tuple(codec.read(store, idx).shape), (2, 3, D))
            self.assertTrue(torch.equal(codec.read(store, idx), full[idx]), codec.name)

    def test_bytes_match_allocation(self):
        for codec in (IdentityCodec(), Int8Codec(BLOCK), Int4Codec(BLOCK), SignCodec(BLOCK)):
            store = codec.allocate(7, GRAD_DIM, DEV)
            n = sum(t.numel() * t.element_size() for t in store.values())
            self.assertEqual(n, 7 * codec.bytes_per_value(GRAD_DIM), codec.name)


class TestCache(unittest.TestCase):
    def _stream(self, n=40, seed=1):
        g = torch.Generator().manual_seed(seed)
        keys = torch.randn(n, 16, generator=g).to(DEV)
        values = grads(n, seed=seed)
        ent = torch.rand(n, generator=g).to(DEV)
        prio = torch.arange(n, device=DEV, dtype=torch.half)
        return keys, values, ent, prio

    def _fill(self, cache, keys, values, ent, prio, batch=7):
        for s in range(0, len(keys), batch):     # batched adds, as Ramen (one add per image)
            for b in range(s, min(s + batch, len(keys))):
                cache.add(keys[b:b + 1], values[b:b + 1], ent[b:b + 1], prio[b:b + 1])

    def test_identity_cache_equals_priority_cache_with_eviction(self):
        keys, values, ent, prio = self._stream()
        a = PriorityCache(9, 16, D, DEV, torch.half)
        b = CompressedPriorityCache(9, 16, D, DEV, torch.half, codec=IdentityCodec())
        self._fill(a, keys, values, ent, prio)      # 40 adds into 9 slots: evicts through argmin (0-d tensor index)
        self._fill(b, keys, values, ent, prio)
        q = torch.randn(5, 16, generator=torch.Generator().manual_seed(2)).to(DEV)
        for k in (1, 3, 9):
            for x, y in zip(a.query(q, k), b.query(q, k)):
                self.assertTrue(torch.equal(x, y), f'topk {k}')
        self.assertTrue(torch.equal(a.priorities, b.priorities))
        self.assertTrue(torch.equal(a.keys, b.keys))

    def test_compressed_cache_keeps_same_entries_and_decodes(self):
        keys, values, ent, prio = self._stream()
        a = PriorityCache(9, 16, D, DEV, torch.half)
        b = CompressedPriorityCache(9, 16, D, DEV, torch.half, codec=Int8Codec(BLOCK))
        self._fill(a, keys, values, ent, prio)
        self._fill(b, keys, values, ent, prio)
        q = torch.randn(5, 16, generator=torch.Generator().manual_seed(3)).to(DEV)
        va, pa, ea, da = a.query(q, 3)
        vb, pb, eb, db = b.query(q, 3)
        self.assertTrue(torch.equal(pa, pb) and torch.equal(ea, eb) and torch.equal(da, db))   # same retrieval
        codec = Int8Codec(BLOCK)                                                                 # values = decode(fp16)
        ref = codec.dequantise(*[t for t in codec.quantise(va.reshape(-1, D))]).reshape(va.shape)
        self.assertTrue(torch.equal(vb, ref))

    def test_nbytes(self):
        for codec in (IdentityCodec(), Int8Codec(BLOCK), Int4Codec(BLOCK), SignCodec(BLOCK)):
            c = CompressedPriorityCache(11, FEAT_DIM, GRAD_DIM, DEV, torch.half, codec=codec)
            self.assertEqual(c.nbytes(), 11 * bytes_per_entry(codec, FEAT_DIM, GRAD_DIM), codec.name)

    def test_identity_nbytes_equals_ramen_cache(self):
        a = PriorityCache(11, FEAT_DIM, GRAD_DIM, DEV, torch.half)
        n = sum(t.numel() * t.element_size() for t in (a.keys, a.values, a.priorities, a.entropies))
        self.assertEqual(n, CompressedPriorityCache(11, FEAT_DIM, GRAD_DIM, DEV, torch.half,
                                                    codec=IdentityCodec()).nbytes())


class TestConfigs(unittest.TestCase):
    def test_blocks_align_with_layernorm_tensors(self):
        self.assertEqual(768 % BLOCK, 0)
        self.assertEqual(GRAD_DIM % BLOCK, 0)
        self.assertEqual(GRAD_DIM, 52 * 768)

    def test_memory_ratios(self):
        self.assertAlmostEqual(memory_ratio('fp16'), 1.0)
        self.assertAlmostEqual(memory_ratio('int8'), (1028 + 39936 + 624 * 4) / (1028 + 79872))
        self.assertLess(memory_ratio('sign'), memory_ratio('int4'))
        self.assertLess(memory_ratio('int4'), memory_ratio('int8'))

    def test_matched_capacity(self):
        for ds, d in DATASETS.items():
            for c in ('int8', 'int4', 'sign'):
                k = matched_capacity(ds, c)
                self.assertLessEqual(abs(k - d['capacity'] * memory_ratio(c)), 0.5 + 1e-9)


if __name__ == '__main__':
    unittest.main()
