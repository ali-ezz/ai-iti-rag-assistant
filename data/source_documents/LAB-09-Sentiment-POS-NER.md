# LAB-09-Sentiment-POS-NER

Source notebook: LAB-09-Sentiment-POS-NER.ipynb

## POS tagging and Named Entity Recognition

Part-of-Speech tagging assigns a grammatical category such as noun, verb, or adjective to
each token. Named Entity Recognition identifies real-world entities and labels categories
such as PERSON, ORGANIZATION, and LOCATION. POS describes a word's grammatical role; NER
describes whether a span refers to a named entity. They are related but separate NLP tasks.

# LAB-9 — IMDB 50K Sentiment + POS/NER (SOLVED, Portable)
**Author: Ali Ezz Ali** | NLP Lec02

ONE self-contained notebook. Runs on **Colab / Kaggle / local** with no local files or absolute paths.

Covers Lec02 fully:
- Sentiment: polarity, lexicon, TextBlob vs VADER vs Transformer
- POS/NER with spaCy en_core_web_sm
- TF-IDF + Logistic Regression baseline
- Multi-model comparison with plots, agreement, confusion matrix, majority vote

### Links
- Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Stanford original: http://ai.stanford.edu/~amaas/data/sentiment/
- HF fallback: https://huggingface.co/datasets/stanfordnlp/imdb
- Model search: https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=sentiment
- distilbert-sst2: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- roberta-twitter: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
- bertweet: https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
- multilingual-student: https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student
- 5-star: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
- spaCy: https://spacy.io/models/en#en_core_web_sm

## 0. Setup — run first

## 1. Imports

## 2. Load IMDB — portable (Colab / Kaggle / local / HF)
Output is always review,sentiment with no manual paths.

## 3. EDA — info, missing, balance, dedup (balanced sample), lengths

## 4. Clean — full run/3 (HTML + URLs) + lecture note (keep raw for Transformer/NER)

## 5. POS Tagging — NLTK + spaCy (batched nlp.pipe)

## 6. NER with spaCy

## 7. VADER (rule-based) — threshold 0.05
Same method as lecture practical.

## 8. TextBlob (lexicon)

## 9. ML Baseline — TF-IDF + Logistic Regression

## 10. Transformers — DistilBERT + RoBERTa + Multilingual + BerTweet

## 11. Consolidated Comparison — all models

## 12. Final Verdict — majority vote + conclusion + save
