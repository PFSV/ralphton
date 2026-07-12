import time

import agent
import data
import tfm

t0 = time.time()
t = data.load(31, "credit-g", True, seed=0)
print("task:", t.name, "ctx", t.X_ctx.shape, "pool", t.X_pool.shape,
      "val", t.X_val.shape, "test", t.X_test.shape, "k", t.n_classes, flush=True)

R = agent.run(t, "agent", 100.0, seed=0, max_iters=4)
print(f"\nbase test AUC {R.test0:.4f} -> final {R.test:.4f} | spent {R.spent} | "
      f"bought {R.rows_bought} synth {R.rows_synth} feats {R.feats_added} llm {R.llm_calls}")
for s in R.steps:
    kept = "KEPT" if s.accepted else "rev"
    print(f"  it{s.it} {s.action:<15s} {s.detail[:46]:<48s} cost={s.cost:<6.2f} "
          f"val {s.val_before:.4f}->{s.val_after:.4f} {kept}")
print("ledger:", R.ledger)
print("tfm calls:", tfm.tfm_calls(), f"| wall {time.time() - t0:.0f}s")
