# M5 — Non-LLM semantic controls (fastText, GloVe, multi-hot)

**Status: COMPLETE.** All four word-vector arrays + the multi-hot control are built, with a
published OOV decision log.
**Date:** 2026-07-13

---

## 1. The plan said this figure was irreproducible. It isn't.

The reproduction plan's risk register rated **MI-12** as R11 and concluded:

> *"the paper's OOV handling was **manual and unlogged**: 'we either corrected the
> misspelling, found a similar word in the fasttext corpus, or removed them', with no
> published list. **Fig. 3a/3c control bars are therefore not exactly reproducible, by
> construction.**"*

That conclusion is wrong, and the evidence is in the release. `word_lists.py:126-362` contains
**`verb_adjustments`**, a 232-entry dictionary mapping each misspelled caption token to either
a correction or one of two drop-sentinels:

```python
verb_adjustments = {
    "unpealed": "_____no_embedding_____",
    "kitboarding": "kite-surfing",
    "staanding": "standing",
    "reahced": "reached",
    ...   # 232 entries
    "widnshield": "_____not_verb_/_unknown_____",
}
```

We parse it out of the release verbatim and apply it exactly as the release does. **Fig. 3a/3c
is reproducible.** The plan's R11 should be struck.

## 2. Artifacts

| Array | Shape | OOV (distinct / occurrences) | Images needing the fallback |
|---|---|---|---|
| `fasttext_categories` | (73000, 300) | **0 / 0** | 0 |
| `fasttext_allwords` | (73000, 300) | 2,234 / 4,524 | 0 |
| `glove_categories` | (73000, 300) | **0 / 0** | 0 |
| `glove_allwords` | (73000, 300) | 2,141 / 2,746 | 0 |
| `multihot` | (73000, 91) | — | — |

Every OOV decision is written to `*_oov_log.json` — **this is the list the original paper does
not publish.**

## 3. Why the category route has zero OOV

Not luck. The release's 91-name list is *pre-adjusted* to the fastText vocabulary
(`word_lists.py:3-17`): fire hydrant → `fireplug`, street sign → `signpost`, parking meter →
`self-parking`, potted plant → `houseplant`, hair drier → `hairdryer`, sports ball → `ball`.
Three categories have no single token at all and are constructed as the mean of two words:

```
baseball-bat   -> mean(baseball, bat)
baseball-glove -> mean(baseball, glove)
tennis-racket  -> mean(tennis, racket)
```

So the "OOV problem" the paper describes was solved *at the vocabulary-design stage* for the
Fig. 3a branch, and only ever bites the Fig. 3c (all-caption-words) branch. The paper's prose
conflates the two.

## 4. Three defects in the released OOV handling (which we log rather than silently fix)

1. **`noun_adjustments` is empty.** Nouns receive no spelling correction at all, while verbs
   get 232 hand-written fixes. The asymmetry is undocumented.
2. **The verb map is applied unconditionally** — including when the encoder is GloVe (a
   different vocabulary, for which the map was not built) and even when it is MPNet, where no
   token can ever be OOV. Applying a fastText-derived misspelling map to MPNet silently alters
   the strings MPNet sees.
3. **Words still OOV after the map are silently dropped** (`KeyError` → `pass`). In the
   `allwords` route the release has **no fallback**, so an all-OOV image would yield
   `np.mean([])` = NaN. We log such images instead (there are none, but the release would not
   have told you).

## 5. Deviation from the paper worth recording

The paper says fastText was used; the release loads `crawl-300d-2M.vec`, the **text** vector
file, not the `.bin` model. That means **fastText's subword OOV inference — its headline
capability — is unavailable**, and OOV words are simply dropped. A reimplementer who loads the
`.bin` model would get *no* OOV at all and a materially different Fig. 3c control bar.

## 6. Dictionary for M11 (decoding)

Google Conceptual Captions fetched: **3,334,173 captions** (train + validation). The paper
states "3.1 million" — a slightly different snapshot or filtering. Recorded, not reconciled.
