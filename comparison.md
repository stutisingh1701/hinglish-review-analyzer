# Classical vs transformer

## Language ID (HingLID test)

| Type | Model | macro_f1 | accuracy |
|---|---|---|---|
| classical | **CRF (word/char features)** | **97.86** | 98.23 |
| classical | HMM bigram + Viterbi | 97.84 | 98.21 |
| classical | CRF + char-LM features | 97.67 | 98.07 |
| classical | Char 4-gram LM (add-k, no context) | 93.39 | 94.64 |

## NER (Singh et al. 2018, same split)

| Type | Model | f1 | precision |
|---|---|---|---|
| classical | **CRF (word/char/context)** | **74.37** | 80.15 |
| classical | CRF + language-ID features | 73.83 | 79.33 |
| classical | HMM bigram + Viterbi | 44.25 | 47.83 |

## Aspect category detection (ACSA-Mix)

| Type | Model | f1 | precision |
|---|---|---|---|
| classical | **Linear SVM** | **85.76** | 87.37 |
| classical | Logistic Regression | 85.00 | 85.92 |
| classical | Complement NB | 77.55 | 84.12 |

## Aspect sentiment (ACSA-Mix)

| Type | Model | macro_f1 | accuracy |
|---|---|---|---|
| classical | **Logistic Regression** | **57.64** | 72.43 |
| classical | Linear SVM | 56.07 | 72.33 |
| classical | Complement NB | 46.94 | 73.64 |
