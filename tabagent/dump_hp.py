import json
from tabicl.prior._prior_config import DEFAULT_SAMPLED_HP
for k, v in DEFAULT_SAMPLED_HP.items():
    if k == "mlp_activations":
        print(f"{k}: <{len(v['choice_values'])} activation factories>")
        continue
    print(f"{k}: {json.dumps(v, default=str)}")
