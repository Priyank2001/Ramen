"""
Execute notebooks in place, one after another, saving after every cell so progress is visible.
Usage: CUDA_VISIBLE_DEVICES=<gpu> python shell/run_notebooks.py nb1.ipynb [nb2.ipynb ...]
"""
import sys
import time

import nbformat
from nbclient import NotebookClient

for path in sys.argv[1:]:
    nb = nbformat.read(path, as_version=4)
    n_code = sum(c.cell_type == 'code' for c in nb.cells)
    done = [0]
    start = time.time()

    def save_progress(cell, cell_index, **kwargs):
        done[0] += cell.cell_type == 'code'
        nbformat.write(nb, path)
        print(f'[{time.strftime("%H:%M:%S")}] {path}: cell {done[0]}/{n_code} done', flush=True)

    print(f'[{time.strftime("%H:%M:%S")}] start {path}', flush=True)
    client = NotebookClient(nb, timeout=None, kernel_name='python3', on_cell_executed=save_progress,
                            resources={'metadata': {'path': 'new_experiments'}})
    try:
        client.execute()
        print(f'[{time.strftime("%H:%M:%S")}] finished {path} in {(time.time() - start) / 60:.1f} min', flush=True)
    except Exception as e:
        print(f'[{time.strftime("%H:%M:%S")}] FAILED {path}: {type(e).__name__}: {str(e)[-800:]}', flush=True)
    finally:
        nbformat.write(nb, path)
