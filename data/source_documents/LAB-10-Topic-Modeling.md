# LAB-10-Topic-Modeling

Source notebook: LAB-10-Topic-Modeling.ipynb

# LAB-10 — BoW, TF-IDF & Topic Modeling (LDA vs LSA)
**Student: Ali Ezz — AI ITI Track**

This notebook combines the best pipeline I studied: BoW/TF-IDF demo (lecture) + full BBC 1000 comparison (4 configs at k=5) + coherence sweep k=2..20.
Dataset: `bbc_news.csv` (1000 articles, title + description).

## 0. Setup

# Bag of Words (BoW)

**Bag of Words (BoW)** is a simple technique used in **Natural Language Processing (NLP)** to convert text into **numerical data** that machine learning models can understand.

The main idea is:

> **Represent a text by counting how many times each word appears.**

It is called a **"Bag" of Words** because the model mainly cares about **which words appear and how often**, not their order.

---

## Example

Suppose we have two sentences:

```text
"I love this movie"
"I love this book"
```

### Step 1: Create the Vocabulary

First, we collect all unique words:

```text
love, this, movie, book
```

### Step 2: Count the Words

We create a vector for each sentence.

| Sentence          | love | this | movie | book |
| ----------------- | ---: | ---: | ----: | ---: |
| I love this movie |    1 |    1 |     1 |    0 |
| I love this book  |    1 |    1 |     0 |    1 |

Each sentence is now represented as numbers.

```text
"I love this movie" → [1, 1, 1, 0]

"I love this book"  → [1, 1, 0, 1]
```

---

## Why Do We Need Bag of Words?

Machine learning models cannot directly understand:

```text
"I love this movie"
```

So we convert the text into numbers:

```text
[1, 1, 1, 0]
```

The model can then use these numerical features for tasks such as:

* Sentiment Analysis
* Text Classification
* Spam Detection
* Document Classification

---

## Important Characteristics

### 1. Word Frequency

BoW counts how many times a word appears.

For example:

```text
"I love this movie. This movie is great."
```

The word `movie` appears **2 times**.

---

### 2. Word Order Is Ignored

These two sentences can produce similar word counts:

```text
"Dog bites man"
"Man bites dog"
```

BoW sees the same words:

```text
dog
bites
man
```

It does **not understand the difference in meaning caused by word order**.

---

## Advantages

* Simple to understand
* Easy to implement
* Fast
* Works well for simple text classification problems

## Disadvantages

* Ignores word order
* Creates a large vocabulary for large datasets
* Does not understand word meaning or context
* Produces many zero values (**sparse matrix**)

---

## BoW in a Sentiment Analysis Project

For a movie-review dataset:

```text
Review
   ↓
Text Cleaning
   ↓
Bag of Words
   ↓
Numerical Features
   ↓
Machine Learning Model
   ↓
Positive / Negative
```

For example:

```text
"I love this movie"
        ↓
Bag of Words
        ↓
[0, 1, 1, 1]
        ↓
Machine Learning Model
        ↓
POSITIVE
```

### Simple Definition

> **Bag of Words converts text into numerical vectors based on the frequency of words, while ignoring the order of the words.**

# TF-IDF

**TF-IDF** stands for **Term Frequency–Inverse Document Frequency**.

It is an NLP technique used to convert text into **numerical values** that machine learning models can understand.

The main idea is:

> **TF-IDF gives a higher score to important words and a lower score to common words.**

---

## Why Do We Need TF-IDF?

With **Bag of Words**, we only count how many times a word appears.

For example:

```text
"I love this movie"
"I love this book"
```

The word **"this"** appears in both sentences, so it gets a count.

But `this` is not very useful for distinguishing between the two sentences.

TF-IDF solves this problem by giving **less importance to words that appear in many documents**.

---

## TF-IDF Components

### 1. TF — Term Frequency

TF measures:

> **How often does a word appear in a document?**

For example:

```text
"I love this movie. I love this movie."
```

The word `movie` appears **2 times**, so it has a higher TF than a word that appears only once.

---

### 2. IDF — Inverse Document Frequency

IDF measures:

> **How rare or common is a word across all documents?**

A word that appears in many documents gets a **lower IDF**.

A word that appears in only a few documents gets a **higher IDF**.

For example:

| Word    | Frequency across documents | Importance |
| ------- | -------------------------- | ---------- |
| the     | Very common                | Low        |
| movie   | Common                     | Medium     |
| amazing | Less common                | High       |

---

## TF-IDF Formula

The basic idea is:

```text
TF-IDF = TF × IDF
```

Where:

* **TF** = Term Frequency
* **IDF** = Inverse Document Frequency

A common IDF formula is:

```text
IDF = log(N / df)
```

Where:

* `N` = total number of documents
* `df` = number of documents containing the word

---

## Example

Suppose we have 3 documents:

```text
Document 1: "I love this movie"
Document 2: "I love this book"
Document 3: "This movie is amazing"
```

The word **"this"** appears in all 3 documents.

Therefore:

```text
"this" → Low IDF
```

The word **"amazing"** appears in only 1 document.

Therefore:

```text
"amazing" → High IDF
```

So TF-IDF considers **"amazing" more informative than "this"**.

---

## Bag of Words vs TF-IDF

| Bag of Words                      | TF-IDF                            |
| --------------------------------- | --------------------------------- |
| Counts word frequency             | Measures word importance          |
| Common words can have high counts | Common words get lower importance |
| Simple                            | More informative                  |
| Uses raw counts                   | Uses weighted values              |

### Simple Example

**Bag of Words:**

```text
the → 10
movie → 5
amazing → 2
```

It only tells us **how many times** the words appear.

**TF-IDF:**

```text
the → 0.05
movie → 0.32
amazing → 0.78
```

It tells us **how important each word is in the documents**.

> **Bag of Words = How often does the word appear?**

> **TF-IDF = How important is the word?**

## 1. Dataset — BBC News 1000 (my run)
I use `description` (main text). `title` adds context in the final cleaning (see friend-5 variant). EDA below shows 1000 rows, no nulls.

## 2. Preprocessing — lower, clean, stopwords, tokenize, stem
Steps: lowercase → remove punctuation → remove stopwords → `word_tokenize` → Porter stemming. This turns `elections/women/says` into clean stems and removes the `say/year` noise as much as possible.

### Dictionary (vocab)

## 3. LDA + BoW (k=5)

LDA + BoW

## 4. LDA + TF-IDF (k=5) — TF-IDF usually lifts LDA a bit

tf-idf

LDA + TF-IDF

## 5. LSA + BoW via SVD (k=5)

bow to matrix

lsa using svd

## 6. LSA + TF-IDF (k=5) — WINNER at k=5, cleanest topics (sport vs politics/war)

lsa & tf-idf

## 7. Final comparison at k=5

final

## 8. Sweep k=2..20 — coherence picks k (LSA likes small k, LDA likes larger k)
Below is the full grid from my best run: 4 configs × k=2..20 with `c_v`. Note `BoW+LSA` peaks at k=2 then drops, `TF-IDF+LDA` climbs to k=19.

### Sweep code + scores

### Coherence dict + best k

## Comparison Graph & Bar Chart

## Conclusion — Ali Ezz, what I learned

1. **BoW counts, TF-IDF weighs.** Same shape, TF-IDF down-weighs `the/say/year` and up-weighs rare topical words.
2. **At k=5 the winner is LSA+TF-IDF ≈ 0.496**, vs LDA+TF-IDF ≈ 0.419, LDA+BoW ≈ 0.411, LSA+BoW ≈ 0.323. TF-IDF helps LSA dramatically (0.323 → 0.496).
3. **Sweep:** LSA best at small k (2-4), LDA best at larger k (11-19). So always sweep + eyeball topics, don't trust one k.
4. **Cleaning is everything:** without extra stoplist `{said,says,say,year,mr}` + `filter_extremes`, every topic is polluted by `say/said/mr`.
5. Most interpretable LSA+TF-IDF topics: `cup/england/world/win/final` = Sport, `police/minister/ukraine/russia/war` = Politics/War.

— Ali Ezz | AI-ITI | LAB-10
