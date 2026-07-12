import numpy as np
import torch.nn as nn

import priortrain as pt

clf = pt.load_backbone(0)
m1 = clf.model_
print("after 1st fit: id(model_)", id(m1))

pt.inject_lora(m1)
n_lora = sum(isinstance(m, pt.LoRALinear) for m in clf.model_.modules())
print("LoRALinear modules after inject:", n_lora)

clf.fit(np.random.randn(64, 5), np.random.randint(0, 2, 64))
m2 = clf.model_
print("after 2nd fit: id(model_)", id(m2), "| same object:", m1 is m2)
print("LoRALinear modules still present:", sum(isinstance(m, pt.LoRALinear) for m in m2.modules()))

# does predict use model_ at all, or a copy?
import inspect
src = inspect.getsource(type(clf).fit)
print("\n--- fit() source (first 40 lines) ---")
print("\n".join(src.splitlines()[:40]))
