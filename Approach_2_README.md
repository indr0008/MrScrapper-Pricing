# Marketplace Price Prediction Challenge — Approach 2

# Shop / Product Level Model

## Overview

This project explores a more personalised marketplace pricing model by directly incorporating:

* `shopId`
* `itemId`
* `modelId`

into the learning process.

Unlike the global marketplace model, this approach attempts to capture:

* shop-specific pricing behaviour
* product-level pricing strategies
* model-level pricing consistency
* entity-specific discount patterns

The motivation behind this approach is:

> Different shops and products may exhibit unique pricing dynamics that cannot be fully captured using only aggregate marketplace features.

---

# Problem Statement

The task is to:

1. train a marketplace price prediction model
2. infer missing prices for future days
3. use daily anchor samples for calibration

The inference data contains:

* only IDs for most products
* 100 anchor rows per day with full information and known prices

---

# Key Difference From Approach 1

## Approach 1

Used:

* engineered historical features only

Did NOT directly use:

* raw IDs

---

## Approach 2

Directly included:

* `shopId`
* `itemId`
* `modelId`

as model inputs.

The hypothesis was:

```text id="a1d0qx"
pricing behaviour is highly entity-specific
```

and therefore:

* explicit entity conditioning may improve performance.

---

# Initial Attempt — XGBoost With Categorical Support

The first implementation used:

## XGBoost Regressor

with:

```text id="8c7h4f"
enable_categorical = True
```

and categorical casting applied to:

* `shopId`
* `itemId`
* `modelId`

The expectation was that:

* XGBoost categorical splitting
* combined with historical features

would capture entity-level behaviour effectively.

---

# Result

Despite enabling categorical handling, performance degraded significantly compared to the global non-ID model.

Observed issues:

* unstable predictions
* weak generalisation
* large errors even for previously seen entities
* inconsistent behaviour on sparse models

In practice:

* the model over-relied on entity splits
* while failing to generalize robustly across the marketplace

---

# Why XGBoost Still Performed Poorly

Even with categorical support enabled, the dataset contains:

* thousands of `itemId`
* tens of thousands of `modelId`

This creates an extremely high-cardinality categorical problem.

Although XGBoost supports categorical splits, it is still not optimized for:

* massive sparse categorical spaces
* marketplace-scale entity encoding
* highly fragmented entity histories

The result was:

* weaker entity generalisation
* unstable partitioning
* poor handling of sparse products

especially compared to CatBoost.

---

# Solution — CatBoost

The pipeline was then migrated to:

## CatBoostRegressor

CatBoost is specifically designed for:

* high-cardinality categorical variables
* target statistics encoding
* entity-aware tree learning
* sparse categorical behaviour

This made it substantially more suitable for:

* marketplace entity modelling
* shop/product-level behaviour learning

---

# Final Model

The final model uses:

* **CatBoostRegressor**
* categorical IDs
* historical marketplace features
* lag features
* trend features
* anchor calibration

---

# Features Used

## 1. Raw Entity IDs

### Categorical Features

* `shopId`
* `itemId`
* `modelId`

These are passed directly as categorical variables.

---

## 2. Historical Features

### Item-Level

* historical average price
* historical min/max price
* volatility
* observation count

### Model-Level

* historical average price
* historical min/max price
* volatility
* observation count

---

## 3. Lag Features

### Item-Level

* previous price
* lag-2 price

### Model-Level

* previous price
* lag-2 price

---

## 4. Trend Features

* previous price change
* trend per day
* directionality

---

## 5. Discount Features

* discount depth
* effective discount percentage
* sharp discount flag

---

## 6. Marketplace Features

* shop popularity
* ratings
* response rate
* review statistics

---

# Training Target

The model predicts:

```text id="t3mcf1"
log1p(price)
```

instead of raw price.

Benefits:

* stabilizes variance
* handles extreme price ranges
* improves tree learning

Predictions are transformed back using:

```text id="6xrybd"
predicted_price = expm1(prediction)
```

---

# Model Performance

## XGBoost With IDs

| Metric         | Result   |
| -------------- | -------- |
| Performance    | degraded |
| Behaviour      | unstable |
| Generalisation | weak     |

Even with:

```
enable_categorical=True
```

XGBoost still underperformed substantially.

---

## CatBoost With IDs

| Metric | Result |
| ------ | ------ |
| RMSE   | ~2.8M  |
| MAE    | ~500K  |
| MAPE   | ~0.64% |

This was a major improvement over:

* XGBoost with categorical IDs

and demonstrated that:

```
the modelling framework itself matters greatly for high-cardinality entity learning
```

---

# Feature Importance Findings

Top features included:

| Feature                    | Importance              |
| -------------------------- | ----------------------- |
| item previous price        | extremely high          |
| lag-2 item price           | high                    |
| model historical max price | high                    |
| model previous price       | high                    |
| shopId                     | meaningful contribution |
| itemId                     | meaningful contribution |

This suggests:

* entity identity does contain predictive value
* but requires appropriate categorical modelling

---

# Key Findings

## 1. Marketplace Prices Are Highly Sticky

Previous prices were by far the strongest predictors.

This indicates:

* low short-term volatility
* strong price persistence

---

## 2. IDs Alone Are Not Sufficient

Simply adding raw IDs:

* does not automatically improve performance

The modelling framework itself is critical.

---

## 3. CatBoost Is Better Suited For Marketplace IDs

CatBoost successfully handled:

* high-cardinality categorical entities
* sparse products
* shop-level behaviour

much better than XGBoost.

---

## 4. Historical Features Still Dominate

Even with IDs:

* lag/history features remained the strongest signals

This confirms:

```
historical pricing behaviour is the core marketplace signal
```

---

# Anchor-Based Daily Calibration

Each future day contains:

* 100 anchor samples with true prices

The pipeline:

1. predicts anchor prices
2. computes daily market shift
3. adjusts all predictions for that day

Daily shift:

```
shift =
actual_price / predicted_price
```

Final adjustment:

```
adjusted_prediction =
raw_prediction × median_daily_shift
```

---

# Inference Pipeline

## Step 1

Load future inference CSV.

---

## Step 2

Expand schema to match training columns.

---

## Step 3

Fill-forward historical features using latest known entity state.

Primary lookup:

```text id="w9cw06"
(shopId, itemId, modelId)
```

Fallback lookup:

```text id="kh7q2u"
itemId
```

---

## Step 4

Generate CatBoost predictions.

---

## Step 5

Compute anchor-based market adjustment.

---

## Step 6

Export final adjusted predictions.

---

# How To Run

## Install Dependencies

```
pip install pandas numpy scikit-learn xgboost catboost
```

---

## Run Notebook

```
catboost_entity_model.ipynb
```

The notebook:

1. engineers features
2. trains XGBoost baseline
3. trains CatBoost entity model
4. evaluates performance
5. performs inference
6. applies anchor calibration

---

# Final Conclusion

Approach 2 demonstrated that:

* entity identity contains valuable pricing information
* high-cardinality categorical handling is critical
* XGBoost categorical support alone was insufficient
* CatBoost performs substantially better for entity-aware marketplace modelling

The final CatBoost system successfully captured:

* shop-specific behaviour
* product-level pricing consistency
* historical marketplace dynamics

while maintaining strong predictive accuracy across future inference days.
