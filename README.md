# Hinglish Review Analyzer

**Aspect-based sentiment analysis and opinion summarization for code-mixed Hindi-English (Hinglish) text.**
Course project, CSEG4034P Computational Linguistics & NLP.

> **Problem.** Indian users write reviews in Hinglish ("camera mast hai but battery jaldi khatam ho jati h"), mixing two languages, two scripts and non-standard spellings. English-only NLP tools break on this. This project builds a full pipeline that normalizes Hinglish, identifies the language of every word, tags entities, finds product aspects, scores the sentiment of each aspect, and summarizes many reviews. It also compares classical statistical models (HMM, CRF, NB, SVM) with transformers (HingBERT, mBERT, MuRIL) on the same data.

## Pipeline

```
raw review ─► 1 normalize ─► 2 tokenize ─► 3 language ID ─► 4 NER ─► 5 aspects ─► 6 aspect sentiment ─► 7 summary
              script → Roman   regex         n-gram LM          CRF       lexicon /       LR/SVM + lexicon       TextRank
              spelling fixes                 HMM / CRF                    SVM (ACD)       (negation-aware)       + counts
```

| Stage | Classical method (this repo) | Transformer (Colab notebook) | Course mapping |
|---|---|---|---|
| Normalization | Rule-based Devanagari→Roman transliterator with schwa deletion, spelling-variant lexicon, elongation fix, emoji tokens | none | Unit I pre-processing · Lab CO1 |
| Tokenization | Regex tokenizer (Devanagari, numbers+units, emoticons) | WordPiece | Unit I · Lab CO1 |
| Language ID | Char 4-gram LM (interpolated, add-k smoothing), **HMM + Viterbi** (from scratch), CRF | token classification | Unit II (HMM, CRF, N-grams) · Lab CO2 (N-gram LM with smoothing) |
| NER | HMM, CRF, CRF + language-ID features | token classification | Unit IV information extraction · Lab CO1 (NER) |
| Aspect category detection | TF-IDF (word + char n-grams) + Complement NB / LogReg / Linear SVM | multi-label sequence classification | Unit II (SVM) · Unit V |
| Aspect sentiment | Aspect-crossed features + NB / LogReg / SVM; Hinglish lexicon with negation and intensifiers | (text, aspect) sentence-pair classification | Unit V sentiment analysis · Lab CO3 |
| Summarization | TextRank (PageRank on TF-IDF cosine graph) + per-aspect counts; ROUGE implemented | none | Unit V opinion summarization · Unit III cosine similarity · Lab CO3 |
| Transformers | none | mBERT, HingBERT, HingMBERT, MuRIL fine-tuning | Unit III pre-trained transformers (BERT) |
| Own corpus | YouTube collection, Code-Mixing Index filter, Label Studio annotation, Cohen's kappa | none | Unit II corpus creation and annotation |
| Evaluation | accuracy, macro-F1, exact-span NER F1, multi-label F1, kappa, Code-Mixing Index | same metrics | all COs |

## Results (classical models, real public data)

Full tables: [`results/results.md`](results/results.md). Classical vs transformer: [`results/comparison.md`](results/comparison.md) (run the notebook to fill the transformer rows).

**1. Word-level language ID** (L3Cube-HingLID test, 6,420 sentences, 194k tokens)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Char 4-gram LM, add-k (no context) | 94.64 | 93.39 |
| HMM bigram + Viterbi | 98.21 | 97.84 |
| CRF (word/char/context) | **98.23** | **97.86** |
| CRF + char-LM features | 98.07 | 97.67 |

**2. NER** (Singh et al. 2018, 464 test sentences, exact-span micro scores)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| HMM bigram + Viterbi | 47.83 | 41.17 | 44.25 |
| CRF (word/char/context) | **80.15** | **69.37** | **74.37** |
| CRF + language-ID features | 79.33 | 69.04 | 73.83 |

**3. Aspect category detection (ACD) and aspect sentiment (ACSC)** (ACSA-Mix test, 777 reviews, romanized text)

| Model | ACD F1 | ACSC Accuracy | ACSC Macro-F1 | Joint (aspect, polarity) F1 |
|---|---|---|---|---|
| Complement Naive Bayes | 77.55 | **73.64** | 46.94 | 57.70 |
| Logistic Regression | 85.00 | 72.43 | **57.64** | **63.04** |
| Linear SVM | **85.76** | 72.33 | 56.07 | 62.91 |

Same models on the original Devanagari + English text: best joint F1 65.03 (Linear SVM).

## Datasets

| Dataset | Used for | Size | Licence | How you get it |
|---|---|---|---|---|
| [L3Cube-HingLID](https://github.com/l3cube-pune/code-mixed-nlp) (Nayak & Joshi, 2022) | Language ID | 31,756 / 6,279 / 6,420 sentences (train/val/test) | CC BY-NC-SA 4.0 | `scripts/download_data.sh` |
| [Hinglish NER tweets](https://github.com/SilentFlame/Named-Entity-Recognition) (Singh et al., 2018) | NER | 3,085 sentences, Person / Org / Location | see repo | `scripts/download_data.sh` |
| [ACSA-Mix](https://github.com/20118/ACSA-Mix) (IEEE TCSS) | Aspect category + sentiment | 3,041 train / 800 test restaurant reviews | see repo | `scripts/download_data.sh` |
| [SemEval-2020 SentiMix](https://zenodo.org/record/3974927) | Sentence sentiment (optional) | ~20k tweets | see Zenodo | manual, then `--sentimix-train/--sentimix-test` |
| POS (Singh et al., 2018 / [LinCE](https://ritual.uh.edu/lince)) | POS tagging (optional) | 1,489 tweets | see source | manual, then `--pos-train/--pos-test` |
| **Own product reviews** | Product-domain evaluation | 300–500 YouTube comments (to collect) | yours | `scripts/collect_youtube.py` → annotate |

ACSA-Mix is mostly Devanagari + English. The pipeline transliterates it to Roman, so a model trained on it can read Roman Hinglish input.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/download_data.sh          # clones the 3 public datasets into data/external/
python scripts/train_all.py            # trains + evaluates everything (~5 min on a laptop CPU)
python tests/smoke_test.py             # end-to-end check
streamlit run app.py                   # demo: single review, batch summary, results
```

Trained models are included in `models/` (scikit-learn 1.9). If loading fails on your machine, re-run `python scripts/train_all.py`.

Use it from Python:

```python
from hinglish.pipeline import HinglishAnalyzer
an = HinglishAnalyzer()
an.analyze("Camera bohot mast hai but battery jaldi khatam ho jati h 😞")["aspects"]
# [{'aspect': 'camera', 'label': 'positive', ...}, {'aspect': 'battery', 'label': 'negative', ...}]
reviews_out, summary, text = an.summarize(list_of_reviews)
```

## Transformers (GPU)

Open `notebooks/transformer_finetune_colab.ipynb` in Google Colab (T4 GPU, free). It runs the classical baselines, then fine-tunes **mBERT, HingBERT, HingMBERT-mixed and MuRIL** on the same four tasks and splits, and writes `results/comparison.md`. About 1 hour for all models × tasks.

Single run: `python scripts/finetune_transformer.py --task acsc --model l3cube-pune/hing-mbert-mixed --fp16`

## Building your own product-review dataset

1. **Collect.** `export YT_API_KEY=...` then `python scripts/collect_youtube.py --query "smartphone review hindi" --videos 15`
2. **Filter.** `python scripts/filter_hinglish.py data/own/raw_comments.csv --keep 500` keeps genuinely code-mixed comments (both languages present, Code-Mixing Index ≥ 10) and pre-suggests aspects.
3. **Annotate.** Load `annotation/label_studio_config.xml` in [Label Studio](https://labelstud.io) and follow [`annotation/guidelines.md`](annotation/guidelines.md). Two people label the same 100 comments.
4. **Export.** `python scripts/labelstudio_to_csv.py export.json` gives one CSV per annotator.
5. **Agreement.** `python scripts/kappa.py data/own/annotated_1.csv data/own/annotated_2.csv` (target kappa ≥ 0.7).
6. **Split and evaluate.** `python scripts/split_dataset.py data/own/annotated.csv` then `python scripts/evaluate_own.py data/own/annotated_test.csv`.

`data/own/seed_examples.csv` holds 40 short product sentences **written by hand for testing the code**. It is not collected data and not a benchmark. Replace it with your annotated YouTube data.

## Findings so far

1. **Language ID: HMM ≈ CRF (97.8 macro-F1), both far above the context-free n-gram LM (93.4).** About 35% of the HMM's 3,461 errors fall on a dozen short words that exist in both languages: *to* (toh), *is*, *the* (were), *me* (in), *do* (two), *sir*. Among the 200 most frequent error patterns, the CRF's wider context fixes 48%. Full list: `results/errors_lid_hmm.csv`.
2. **NER: the CRF beats the HMM by 30 F1 points (74.4 vs 44.3).** The HMM only sees word identity, so it cannot use capitalisation, hashtags or suffixes on unseen names. Adding language-ID features to the CRF did not help (73.8).
3. **Aspect sentiment: accuracy is 72–74% but macro-F1 only ~57%.** *Conflict* and *neutral* are rare (695 of 3,708 training pairs), and the biggest error group is positive↔negative confusion inside multi-aspect sentences (102 of 274 errors, `results/errors_acsc_lr.csv`).
4. **Romanizing Devanagari costs little:** best joint F1 drops from 65.0 to 63.0. In return, the restaurant-trained models can read Roman Hinglish input.
5. **Domain shift.** On the 40 hand-written phone sentences, aspect detection reaches 96.5 F1 but sentiment accuracy is only 52%. A sentiment model trained on restaurant reviews plus a small lexicon does not transfer to phone reviews ("battery jaldi khatam" is negative only for products). This is why the project collects and annotates its own product reviews and fine-tunes transformers on them.

**Not done yet / limitations:** transformer numbers need a GPU run (notebook). POS tagging code is ready, but the POS data must be downloaded manually. The NER split is a random 85/15 split, not the official LinCE split. Sarcasm is not handled.

## Project structure

```
hinglish/            core library
  normalize.py       transliteration, spelling normalization, tokenizer, clause splitter
  hmm.py             bigram HMM tagger + Viterbi (from scratch)
  crf_tagger.py      CRF features (word, affixes, shape, context window)
  langid.py          char n-gram LM LID, CRF LID, Code-Mixing Index
  absa.py            aspect category detection, aspect sentiment, lexicon sentiment, product aspect extractor
  summarize.py       TextRank + opinion summary
  metrics.py         token/span/multi-label metrics, ROUGE
  pipeline.py        HinglishAnalyzer (end-to-end)
  data.py            dataset loaders (HingLID, NER, ACSA-Mix, SentiMix, CoNLL, own CSV)
scripts/             download, train/evaluate, transformer fine-tuning, dataset tools
data/lexicons/       spelling variants, Devanagari exceptions, sentiment lexicon, product aspects
annotation/          Label Studio config + guidelines
notebooks/           Colab notebook for transformers
results/             results.md, comparison.md, error files
app.py               Streamlit demo
```

## References

1. Vyas et al. (2014). POS Tagging of English-Hindi Code-Mixed Social Media Content. EMNLP.
2. Das & Gambäck (2014). Identifying Languages at the Word Level in Code-Mixed Indian Social Media Text. ICON. (Code-Mixing Index)
3. Akhtar, Ekbal & Bhattacharyya (2016). Aspect based Sentiment Analysis in Hindi: Resource Creation and Evaluation. LREC.
4. Singh, Sen, Kumaraguru (2018). A Twitter Corpus for Hindi-English Code Mixed POS Tagging. SocialNLP.
5. Singh, Vijay, Akhtar, Shrivastava (2018). Named Entity Recognition for Hindi-English Code-Mixed Social Media Text. NEWS.
6. Aguilar, Kar & Solorio (2020). LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation. LREC.
7. Khanuja et al. (2020). GLUECoS: An Evaluation Benchmark for Code-Switched NLP. ACL.
8. Patwa et al. (2020). SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets.
9. Nayak & Joshi (2022). L3Cube-HingCorpus and HingBERT. WILDRE @ LREC.
10. Atmosphere kamaal ka tha (was wonderful): A Multilingual Joint Learning Framework for Aspect Category Detection and Sentiment Classification. IEEE TCSS (2024). (ACSA-Mix)
11. Mihalcea & Tarau (2004). TextRank: Bringing Order into Texts. EMNLP.
