"""Probe the suite once (sequentially -- concurrent OpenML downloads corrupt its cache),
pin the tasks that actually load, then materialise the per-seed splits."""
import sys, time
import tabarena

if __name__ == "__main__":
    print("=== probing TabArena suite (sequential) ===", flush=True)
    t0 = time.time()
    ok, bad = tabarena.probe_suite()
    print(f"probe took {time.time()-t0:.0f}s\n", flush=True)

    for seed in (0, 1, 2):
        t0 = time.time()
        dev, test = tabarena.load_split(seed)
        print(f"seed {seed}: DEV {len(dev)} / TEST {len(test)} cached in {time.time()-t0:.0f}s",
              flush=True)
