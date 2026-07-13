# M12 — Figure 2a: functional contrasts from novel sentences

**Status: REPRODUCED, with a serious qualification the paper's own method conceals.**
**Date:** 2026-07-13

---

## 1. What this figure is

Fifteen hand-written sentences (5 People, 5 Places, 5 Food) are pushed through the **frozen**
encoding model, the 5 predicted maps within a category are averaged, and the categories are
contrasted. The claim (C4) is *topographic*: People should land on face/body areas (FFA, OFA,
EBA, pSTS), Places on scene areas (PPA, OPA), Food on Pennock et al.'s food-selective regions.

The 15 sentences are reproduced verbatim from the release (`encoding_decoding_utils.py:191-216`),
so this analysis is exactly reproducible *as run*, even though the authors state "we did not
have a precise method for selecting these sentences."

**This is the only analysis in the paper with no multiple-comparison correction.** The paper
says so explicitly, and the release confirms it (`text_to_brain_prediction.py:80`,
`sig_mask='uncorrected'`, p < 0.05 two-tailed across the 8 subjects), while the
encoding-accuracy map right next to it uses `fdr_bh`.

We reproduce it **as published** (uncorrected) and additionally compute the FDR-corrected
version, because that is the first question any reviewer would ask.

## 2. Result

| contrast | significant vertices, **uncorrected (as published)** | **with FDR** | ratio |
|---|---|---|---|
| People − Places | **41,666** | **714** | 58× |
| Food − People | **32,652** | **0** | ∞ |

### People − Places: topography reproduces

| ROI | contrast |
|---|---|
| **midventral** | **+0.351** |
| **midlateral** | **+0.213** |
| **lateral** | **+0.206** |
| ventral | +0.049 |
| early | −0.059 |
| parietal | −0.146 |
| midparietal | −0.151 |

People > Places in midventral / midlateral / lateral — where FFA, OFA, EBA and pSTS live.
Places > People in parietal / midparietal — consistent with OPA. **This is C4's prediction, and
it holds.** It also survives FDR, though 58× thinner than the published map.

### Food − People: does not survive its own missing correction

| ROI | contrast |
|---|---|
| midparietal | +0.235 |
| midlateral | +0.131 |
| parietal | +0.134 |
| early | +0.123 |
| **ventral** | **+0.001** |
| midventral | −0.064 |
| lateral | −0.200 |

**Zero vertices survive FDR.** And the ROI profile is wrong for the claim: **ventral is +0.001**
— essentially nothing — where the paper predicts food-selective ventral cortex (Pennock et al.
2021). The strongest positive lobes are in midparietal and early visual cortex, which is not
what a food-selectivity claim wants.

## 3. Verdict on C4

**REPRODUCED for people/places. NOT SUPPORTED for food.**

The paper's decision to omit FDR from precisely this one figure is doing real work: with the
correction it itself applies everywhere else, the People−Places map thins by 58× and **the
Food−People map disappears entirely**.

This does not mean the authors did anything hidden — they state the omission plainly, and they
present the figure as a qualitative demonstration rather than a test. But it does mean the
figure cannot carry more weight than a qualitative demonstration, and the food component
cannot carry any weight at all. The reproduction plan already rated C4 "the weakest claim in
the paper on evidentiary grounds" (R8/H4). That rating is confirmed, and can now be made
quantitative.

## 4. Caveats

- Our splits, ridge fraction, and encoding model are ours, not the authors'. The *published*
  maps may differ in detail. What is robust is the **direction** of the effect: applying the
  paper's own standard correction to its own uncorrected figure removes essentially all of the
  food evidence.
- The contrast is a difference of *predicted* voxel activity, so it inherits every assumption
  of the encoding model (hidden assumption H4).
- `frac = 0.05` for all 8 subjects — the same inert-regularization result found in Fig. 1.
