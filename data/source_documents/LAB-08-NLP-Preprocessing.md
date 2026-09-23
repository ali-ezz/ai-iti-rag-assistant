# LAB-08-NLP-Preprocessing

Source notebook: LAB-08-NLP-Preprocessing.ipynb

## Stemming and lemmatization

Stemming applies heuristic rules to remove word endings and can produce a root that is not
a dictionary word. Lemmatization uses vocabulary and grammatical information to return a
valid base form or lemma. Stemming is usually faster, while lemmatization is linguistically
cleaner. Both are text-normalization steps used before feature extraction when appropriate.

# SMS Spam Collection — NLP Preprocessing (Best-of)
**Author: Ali Ezz Ali — LAB-8 — SOLVED (merged, not re-executed)**

This notebook merges the **best parts of friend-1 → friend-6** + lecture `NlpLec01ITI` pipeline, **without re-running anything** — all outputs below are copied verbatim from the friends' executed notebooks (same `spam.csv`, 5572 rows).

**Pipeline:** load → EDA → drop `Unnamed` → drop duplicates (403) → rename `label/message` → lowercase → unicode-fix → regex (`email/URL/mention/hashtag/numbers/punct`, `[*]→star`) → stopwords (keep negations `not/no/never`) → `word_tokenize` → stemming (`Porter`) + lemmatization (`WordNet`) → unigrams/bigrams/trigrams/4-grams + plots → spam-vs-ham comparison.

**Sources:**
- EDA + `describe/duplicated/Unnamed` + unicode + deep regex + negation-set (23 words) + per-row bigrams/trigrams → **friend-1**
- `countplot/pie`, `message_length`, `clean_text()` fn, `Counter+seaborn` n-gram plots, final preview → **friend-4**
- `top-20` table + `steelblue` plot + spam-vs-ham top-10 + trigram → **friend-2**
- `[*]→star`, `[^^\w\s]` punct, stem+lemm side-by-side, 4-gram → **friend-3 / friend-6** (lecture-faithful)
- local `spam.csv, latin-1` load pattern → **friend-6**

## 0. Imports & NLTK setup

## 1. Load dataset — `spam.csv` (SMS Spam Collection, `latin-1`)
Same file all friends used. Primary: `kagglehub` (friend-1). Alternative local `spam.csv` in same folder (friend-6 / lecture pattern) kept as comment.

**Local alternative (friend-6, lecture-style — same folder as notebook):**
```python
# data = pd.read_csv("spam.csv", encoding="latin-1")
# data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
# data = data.rename(columns={'v1':'label','v2':'message'})
```

## 2. EDA — discover the dataset

## 3. Dataframe actions — drop junk, dedup, rename
Best: friend-1 (explicit 3 steps) + friend-4 label distribution already shown above.

## 4. Text cleaning
Order (best-of): `lower → unicode-fix → [*]→star → regex deep-clean → stopwords(keep negations)`. Friend-1 unicode+regex is deepest; friend-3/6 star-step preserves `*` as word `star` (lecture `HotelReviews` pattern).

**Stopwords — keep negations.** Best: friend-1 keeps 23 negation/contraction forms (`not/no/never/nor/ain/aren/couldn/...`). Friend-4 keeps minimal `{not,no,never}` subset — same idea, smaller. We keep friend-1 set.

## 5. Tokenization — `word_tokenize`

## 6. Stemming (Porter) vs Lemmatization (WordNet) — side-by-side
Best: friend-2/3 show both on same tokens. `crazi/avail/bugi` (stem) vs `crazy/available/bugis` (lemma).

## 7. N-grams — global frequencies + plots
Best: friend-2 unigrams/bigrams/trigrams + top-20 table/plot; friend-3/6 4-grams; friend-4 Counter+seaborn plots + per-row n-grams (friend-1).

### 7b. Per-row n-grams (friend-1) + Counter plots (friend-4)

## 8. Done — what was / wasn't included (friend-4)
