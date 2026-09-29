# Product Review Sentiment Analysis: Brief Write-up

**Author:** Tanay Chothani, BBA Artificial Intelligence, Symbiosis Artificial Intelligence Institute, Pune
**Notebook:** `sentiment_pipeline_with_graphs.ipynb`

## 1. Problem

The goal is to classify clothing reviews as **positive** or **negative**. Reaching about 90% accuracy is easy, so the project asks a harder question: *does one sentiment model treat every product category and every kind of review fairly?* The project has three parts:

1. **Build** a working sentiment classifier.
2. **Audit** it by product category and by review length, not just with one overall score.
3. **Explain** which words drive its predictions.

Reviews often praise a product and then complain ("loved the look, **but** it fell apart"). The project also tests a small fix aimed at this pattern.

## 2. Dataset

**Women's E-Commerce Clothing Reviews**, a public dataset released by Nick Brooks.

| Item | Value |
|---|---|
| Raw reviews | 23,486 |
| Used after cleaning | 19,818 |
| Positive / negative | 88.0% / 12.0% |
| Median review length | 58 words |
| Train / test split | 15,854 / 3,964 (stratified 80/20; 474 negative, 3,490 positive in test) |

**Labels** come from star ratings: 4–5★ is positive, 1–2★ is negative, and 3★ reviews are dropped as ambiguous.

## 3. Preprocessing

- Drop reviews with no text and all 3-star reviews.
- Merge categories with fewer than 50 reviews into "Other".
- Split reviews into **short** and **detailed** at the median word count (58).
- Clean the text: lowercase, remove punctuation and digits, collapse whitespace.
- **Contrast-aware variant only:** the clause after the first contrast word (*but, however, although, though, unfortunately, except, yet*) is duplicated, roughly doubling its weight. Example: "loved the look but after one wash it shrank terribly" becomes "… **after one wash it shrank terribly**".

## 4. Approach and model

Two variants share the same settings and the same train/test split, so the comparison is fair:

- **Features:** TF-IDF on unigrams and bigrams, English stop-words removed, `min_df=5`, `max_df=0.9`, at most 20,000 features (14,944 kept).
- **Classifier:** Logistic Regression with `class_weight="balanced"`, `max_iter=1000`, `random_state=42`.
- **Baseline:** cleaned text only.
- **Contrast-aware:** the same pipeline with the clause-duplication step added.

A linear model was chosen on purpose. Every prediction is a weighted sum over words, so each word's contribution can be read directly for the explainability and audit steps.

## 5. Results

**Overall (test set, n = 3,964)**

| Metric | Baseline | Contrast-aware |
|---|---|---|
| Accuracy | 0.915 | 0.915 |
| Precision (positive) | 0.979 | 0.979 |
| Recall (positive) | 0.923 | 0.923 |
| F1 (positive) | 0.950 | 0.950 |
| **Recall (negative)** | **0.852** | **0.857** |
| Precision (negative) | 0.602 | 0.601 |
| ROC AUC | 0.961 | 0.961 |

**Category audit (negative-review recall).** Recall varies widely even though overall accuracy looks fine.
- Weakest: Jeans 54% and Pants 64% (68% with contrast-aware).
- Strongest: Sweaters 97% and Knits 90% (87% with contrast-aware).

**Length audit.** Short and detailed reviews differ by at most 3.3 points, so review length is not the main source of bias. Negative recall is 85.6% (baseline) and 87.3% (contrast-aware) on short reviews, and 84.9% and 84.0% on detailed reviews.

**Explainability.** The top positive words are *love, perfect, little, comfortable, great*. The top negative words are *disappointed, cheap, returned, unflattering, huge*. These are genuine sentiment words, not brand or category shortcuts. Words such as "returned" and "huge" lean negative, which suggests the model has partly learned sizing and return language. On the sample review "I loved the look and the fit was great, but after one wash it shrank terribly", the contrast-aware model predicts **negative** (p(positive) = 0.24).

**Contrast-aware fix.** It catches 2 more of the 474 negative reviews (406 vs 404). This is a small gain, not a solution. It helps Pants and Dresses but lowers Knits by about 3 points.

## 6. Limitations

1. **Uneven performance by category.** The model catches complaints about Jeans (54%) and Pants (64%) far less reliably than about Sweaters (97%). Overall accuracy of 91.5% hides this, and the model should not be trusted equally across categories.
2. **Small samples behind the gap.** Jeans has only 13 negative test reviews (7 caught, 95% CI roughly 29–77%) and Pants has 22. The gap is a real warning sign but a noisy measurement, and the same applies to the contrast-aware gain.
3. **Many false alarms on negatives.** Negative precision is about 0.60, so roughly 4 in 10 reviews flagged as negative are actually positive.
4. **Word-counting model.** TF-IDF cannot handle negation ("not soft"), sarcasm, or sentence structure. The contrast-aware trick only approximates the meaning of "but".
5. **Labels and scope.** Sentiment labels are proxies taken from star ratings. There is no neutral class, and the data is one English-language, women's-clothing dataset. Reviewer demographics were not audited, and the evaluation is a single split with no cross-validation.

**Recommended use:** a first-pass filter with human review, tuned toward recall for weak categories such as Jeans and Pants. It should not drive automated actions.

**Future work:** clause-level parsing, negation-scope detection, and a transformer model such as BERT.
