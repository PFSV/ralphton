import time, torch, numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

print("torch", torch.__version__, "| mps:", torch.backends.mps.is_available())
X, y = make_classification(n_samples=800, n_features=12, n_informative=6, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

for dev in (["mps"] if torch.backends.mps.is_available() else []) + ["cpu"]:
    try:
        t0 = time.time()
        clf = TabICLClassifier(device=dev, random_state=0)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        print(f"{dev}: AUC={roc_auc_score(yte,p):.4f}  time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"{dev}: FAILED -> {type(e).__name__}: {str(e)[:200]}")
