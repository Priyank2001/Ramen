"""
CLI: run the shift decomposition on one dataset (CPU only, threads capped).

    python run_shift_decomposition.py ImageNetC5K            # full run, cached in results/
    python run_shift_decomposition.py EuroSAT --smoke        # sanity checks only, no bootstrap / cross-fit / cache
"""
import argparse
import os

THREADS = '16'
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = THREADS

from experiment import ShiftDecomposition  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()
    e = ShiftDecomposition(a.dataset, a.seed, log=lambda s: print(s, flush=True))
    if a.smoke:
        e.B = 10
        e.save = lambda: None
        e.run(use_cache=False, crossfit=False, control=False)
        from attribution import domain_avg
        for n in e.names:
            print(f'{n:12s} {domain_avg(e.correct[n], e.d.dom):6.2f}')
        return
    e.run(use_cache=not a.no_cache)


if __name__ == '__main__':
    main()
