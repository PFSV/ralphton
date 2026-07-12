import warnings, openml, numpy as np
warnings.filterwarnings("ignore")

# semantic = meaningful column names; nonsem = anonymized/opaque features (control arm)
CANDIDATES = {
 "semantic": [31,        # credit-g
              1461,      # bank-marketing
              1494,      # qsar-biodeg
              1590,      # adult  (MEMORIZED per Bordt -> expect to drop)
              40994,     # climate-model-simulation-crashes
              1487,      # ozone-level-8hr
              23,        # cmc (contraceptive)
              1049,      # pc4 software defect
              44,        # spambase
              40701,     # churn
              ],
 "nonsem":  [1489,       # phoneme  (V1..V5)
             1063,       # kc2
             1067,       # kc1
             40983,      # wilt
             1462,       # banknote
             1053,       # jm1
             ],
}
for arm, ids in CANDIDATES.items():
    for did in ids:
        try:
            d = openml.datasets.get_dataset(did, download_data=True,
                    download_qualities=False, download_features_meta_data=True)
            X, y, _, names = d.get_data(target=d.default_target_attribute)
            n, m = X.shape
            k = len(np.unique(y.astype(str)))
            cols = ", ".join(list(X.columns)[:6])
            print(f"{arm:8s} id={did:<6d} {d.name[:28]:30s} n={n:<6d} m={m:<3d} k={k}  cols[{cols}]")
        except Exception as e:
            print(f"{arm:8s} id={did:<6d} FAILED {type(e).__name__}: {str(e)[:70]}")
