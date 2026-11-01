# PC2: secondary server (hostname `test-Auk`)

- Branch: `pc2_lab` (from 2026-11-01; earlier: `sec_server_dual_agent`). Run-id prefix: `pc2_`.
- GPUs: 0 = RTX 3080 Ti 12 GB, 1 = RTX 2080 Ti 11 GB. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Python: `~/priyank/Projects/RAMEN_ENV/bin/python` with `PYTHONNOUSERSITE=1`. There is no `python` on PATH.
- No tmux/screen. Detach long jobs with `setsid nohup ... &`.
- Host: 125 GB RAM, **2 GB swap**, 32 cores. Running out of RAM freezes the server (D17). Use one run per GPU, at most 4 DataLoader workers, and no host-RAM gradient caches.
- Large caches don't fit at batch 100 on 12 / 11 GB, so `pc2_lab` uses a smaller fixed batch per dataset (D18), not host-RAM caches.
- Hard RAM cap: `systemd-run --user --scope -p MemoryMax=<N>G -p MemorySwapMax=0 <cmd>` works (cgroup v2, memory controller delegated). `Linger=no`, so scopes die at logout until the user runs `loginctl enable-linger`.
- Git remote: `personal` (SSH key set per repo via `core.sshCommand`).
