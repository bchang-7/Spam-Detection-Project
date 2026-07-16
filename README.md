# SMS Spam Classifier

A text classification pipeline that detects spam SMS messages, comparing classical
machine learning approaches on the UCI SMS Spam Collection dataset.

## Results

Best model: **linear SVM** with TF-IDF features — **0.92 spam F1** on a held-out test set.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| SVM | 0.951 | 0.885 | **0.917** |
| Balanced SVM | 0.902 | 0.908 | 0.905 |
| Balanced LogReg | 0.868 | 0.901 | 0.884 |
| Naive Bayes | 1.000 | 0.725 | 0.841 |
| LogReg | 0.968 | 0.702 | 0.814 |

*Metrics are for the spam class on a stratified held-out test set (n=1,034).*

While Naive Bayes achieves perfect precision, it misses 27% of spam. Linear SVM offers the
best overall balance, catching 89% of spam while flagging only 6 legitimate messages
out of 903. Since wrongly blocking a real message is costlier than letting spam through, near-perfect
precision matters, but SVM delivers it at a much higher recall, making it a stronger deployment choice.

## Dataset

UCI SMS Spam Collection

- **~5,574 messages** (5,169 after removing 403 duplicates)
- **87% ham / 13% spam** — heavily imbalanced
- [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

## Approach

1. Used EDA to validate feature signals
2. Cleaning: 
    1. Turned all strings into lowercase
    2. Stripped punctuation from strings
    3. Removed stopwords from strings
    4. Porter Stemming to collapse variants (win/winner/winning -> win) so that signal concentrates to one feature rather than split across 3 features
3. Stratified 80/20 split, preserving 87% ham / 13% spam ratio
4. TF-IDF vectorization (fit on training data only)
5. Model comparison: Naive Bayes, Logistic Regression, linear SVM (plain + class-balanced variants)

**Leakage prevention:** the vectorizer learns vocabulary and IDF weights from training data only. Fitting on the full dataset would let test information bleed into the features and inflate scores.

**F1 over accuracy:** with 87% ham, a model predicting "ham" for every message scores 87% accuracy while catching zero spam. Accuracy is misleading on imbalanced data, so spam-class F1 is the headline metric.

## Key Findings

- 95% of spam contains a digit compared to only 16% of ham making it the strongest single feature
- On average, spam averages 139 characters while ham averages 71. This is coupled with the fact that spam messages are less variable with a standard deviation of 29 compared to ham's standard deviation of 58 which reflects the templated and formulaic content of most spam messages
- 34% of spam contains a currency symbol compared to 0.4% of ham making it a strong indicator of spam
- The main vocabulary of spam (including words like call, free, txt, mobile) is entirely absent from ham's top 20 words which is mainly filled by pronouns. This is what motivates TF-IDF

## Project Structure

```
spam-classifier/
├── data/
│   └── raw/                    # UCI SMS Spam Collection
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory analysis
│   ├── 02_preprocessing.ipynb  # cleaning, split, TF-IDF
│   └── 03_modeling.ipynb       # model training and comparison
├── requirements.txt
└── README.md
```
## Setup

1. **Clone the repo:** `git clone https://github.com/bchang-7/Spam-Detection-Project` and `cd` into it
2. **Create and activate a virtual environment:** note the platform difference:
    - Windows (PowerShell): `python -m venv venv` then `venv\Scripts\Activate.ps1`
    - Mac/Linux: `python -m venv venv` then `source venv/bin/activate`
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Run the notebooks in order:** 01 -> 02 -> 03

Note: on the first run, nltk needs its stopwords data via `nltk.download('stopwords)`