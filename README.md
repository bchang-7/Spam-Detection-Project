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

Naive Bayes achieves perfect precision but misses 27% of spam. Linear SVM offers the
best overall balance — catching 89% of spam while flagging only 6 legitimate messages
out of 903.