# LAB-11-Arabic-NLP

Source notebook: LAB-11-Arabic-NLP.ipynb

# LAB-11 — Arabic News: Summarizer + TF-IDF → Sentiment, Topics, ML, Transformers
**Student: Ali Ezz — AI ITI Track**

Data: `summarizdataset.csv` — 8378 Tunisian Arabic news. Columns: `text | type | Processed Text | summarizer`.
Working text = `summarizer` (short, clean, per instructor photo). Label = `type` (9 classes, imbalanced).

Order per photo: **Visualization محترم → Preprocessing → 1) Sentiment 2) Topic Modeling 3) ML Models (3, compare, Kaggle tricks) 4) Transformers — all with TF-IDF where applicable.**
Run in Colab (GPU optional for transformers): upload this ipynb + `summarizdataset.csv` in same folder → Run All.

## 0. Setup

## 1. Load data

## 2. Visualization محترم (EDA)
Known distribution (8378): localnews 3796, internationalNews 1174, sport 1028, society 920, politic 587, diverse 437, economy 261, culture 97, technology 78. Code below recomputes + plots — run to get figures.

### Top words (overall + per type)

## 3. Preprocessing Arabic
Why ≠ English: no Porter. Steps I do: remove diacritics/tatweel, normalize `أإآ→ا ة→ه ى→ي`, remove URLs/numbers/punct (keep Arabic letters), collapse spaces, remove Arabic stopwords, drop len<=1. `Processed Text` col in file shows same idea — compare with it.

## 4. TF-IDF (used by topics + ML) — Kaggle tricks included

## 5. Task 1 — Sentiment Analysis
News `type` is topic, not sentiment — so I demo sentiment on `summarizer` sample with CAMeLBERT-sentiment (Colab). Expect mostly neutral/objective news. This satisfies the photo's task 1 without faking labels.

## 6. Task 2 — Topic Modeling (LDA-BoW vs LSA-TFIDF, coherence picks k)

## 7. Task 3 — ML Models (3, TF-IDF, compare like photo asks)

## 8. Task 4 — Transformers (vs TF-IDF)

## Conclusion — Ali Ezz
1. Work text = `summarizer`, label = `type` (imbalanced: localnews 45%). 2. Arabic cleaning = normalization, not stemming. 3. Sentiment on news = mostly neutral (demo, not fake labels). 4. Topics: expect sport (مباراه/فريق/هدف) vs politic (حكومه/انتخابات) vs economy (اسعار/سوق). Coherence picks k. 5. ML: TF-IDF(1,2)+sublinear+balanced → compare 3, LinearSVC usually wins on news. 6. Transformers (MiniLM/CAMeLBERT) vs TF-IDF = final comparison per photo.
— Ali Ezz | AI-ITI | LAB-11
