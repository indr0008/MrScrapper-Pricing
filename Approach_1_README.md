# Marketplace Price Prediction Challenge

## Overview

This project builds a global marketplace-wide price prediction model capable of estimating product prices across all shops, items, and models using historical marketplace behaviour.

The approach follows:

## Approach 1 — Global Marketplace Model

A single machine learning model is trained on the entire historical dataset to learn general pricing behaviour across the marketplace.

The model:

* generalises across shops, products, and categories
* predicts prices using historical behavioural features
* uses anchor samples to calibrate daily marketplace drift

The implementation intentionally avoids directly using:

* `shopId`
* `itemId`
* `modelId`

as raw model inputs.

Instead, entity behaviour is represented through engineered historical statistics and lag features.

---

# Final Model

The final baseline model uses:

* **XGBoost Regressor**
* log-price regression (`log1p(price)`)
* marketplace-wide feature engineering
* historical item/model behavioural features

---


# Problem Setup

The inference dataset contains:

* a small anchor subset with known prices
* many rows containing only IDs

The task is to:

1. reconstruct useful historical features
2. predict unknown prices
3. optionally calibrate predictions using anchor samples

---

# Feature Engineering

The model relies heavily on historical marketplace behaviour.

## 1. Historical Price Features

### Item-Level

* average historical price
* min historical price
* max historical price
* historical volatility
* observation count

### Model-Level

* average historical price
* min historical price
* max historical price
* historical volatility
* observation count

---

## 2. Lag Features

### Item-Level

* previous observed price
* lag-2 price

### Model-Level

* previous observed price
* lag-2 price

These became the strongest predictive signals.

---

## 3. Trend Features

* previous price change
* price change percentage
* trend per day
* price direction

---

## 4. Discount Features

* discount amount
* discount depth
* effective discount percentage
* sharp discount indicator

---

## 5. Marketplace Features

* shop popularity
* follower count
* ratings
* response rate
* review score
* comment count

---

## 6. Temporal Features

* day of week
* month
* weekend flag
* time-of-day indicators

---

# Model Training

## Target

The model predicts:

```text
log1p(price)
```

instead of raw price.

This stabilizes:

* variance
* extreme prices
* marketplace outliers

Final predictions are transformed back using:

```text
predicted_price = expm1(prediction)
```

---

# Validation Strategy

Validation was performed using:

* a full unseen future day split

This better simulates:

* real-world deployment
* future marketplace inference

The validation split also acts as:

* the anchor simulation set

---

# Model Performance

## XGBoost Global Marketplace Model

| Metric | Result          |
| ------ | --------------- |
| RMSE   | ~1.2M           |
| MAE    | ~237K           |
| MAPE   | ~0.32%          |


Overall performance was very strong.

---

# Key Findings

## 1. Historical Prices Are Extremely Predictive

The most important features were:

| Feature                        | Importance |
| ------------------------------ | ---------- |
| previous item price            | very high  |
| lag-2 item price               | high       |
| historical model max price     | high       |
| historical model average price | high       |

This suggests:

* marketplace prices are highly stable
* most products exhibit strong price persistence

---

## 2. Lag Features Dominate

Removing lag/history features significantly degraded performance:

| Model                        | MAPE   |
| ---------------------------- | ------ |
| Full historical model        | ~0.26% |
| Without lag/history features | ~7.98% |

This confirms:

* historical pricing behaviour is the core predictive signal

---

## 3. Marketplace Drift Is Small

Anchor-based calibration showed:

* minimal daily marketplace shift
* highly stable pricing behaviour

Median daily shift remained very close to:

```text
1.0
```

across validation days.

This indicates:

* the global model already generalizes well
* marketplace-wide drift is limited

---

# Why Build A Model Instead Of Using Last Price?

A simple baseline was considered:

```text
Use previous observed price directly
```

This baseline performs surprisingly well because:

* many products maintain stable prices

However, the ML model still provides important value:

## 1. Handles Missing History

Some products lack recent observations.

The model can still infer prices using:

* marketplace patterns
* historical aggregates
* related behavioural signals

---

## 2. Handles Scraping Errors

Marketplace scraping occasionally produces:

* incorrect prices
* missing values
* corrupted entries

The model can recover reasonable estimates even when raw scraped values are unreliable.

---

## 3. Handles True Price Changes

Although infrequent, real price changes do occur.

The model captures:

* discounts
* promotions
* pricing trends
* marketplace dynamics

better than simple carry-forward logic.

---

# Anchor-Based Calibration

Each inference day contains:

* 100 anchor samples with known prices

The pipeline:

1. predicts anchor prices
2. compares prediction vs actual
3. estimates daily marketplace shift
4. adjusts all predictions for that day

Daily adjustment:

```text
adjusted_prediction =
raw_prediction × daily_market_shift
```

Median shift is used for robustness.

---

# Inference Workflow

## Step 1

Load incoming inference CSV.

---

## Step 2

Expand missing columns to match training schema.

---

## Step 3

Fill-forward historical features using latest known entity state.

Primary lookup:

```text
(shopId, itemId, modelId)
```

Fallback lookup:

```text
itemId
```

---

## Step 4

Generate predictions using XGBoost.

---

## Step 5

Use anchor samples to estimate daily shift.

---

## Step 6

Adjust final predictions.

---

# How To Run

## Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost
```

---

## Run Notebook

```bash
xgboost_global_model.ipynb
```

The notebook:

1. builds features
2. trains model
3. evaluates performance
4. performs inference
5. applies anchor calibration

---

# Final Conclusion

The global marketplace model successfully learned:

* marketplace pricing structure
* historical pricing behaviour
* discount effects
* temporal patterns

The strongest predictive signals were:

* historical prices
* lag features
* historical entity behaviour

The final system:

* generalizes well
* handles sparse observations
* supports anchor-based daily calibration
* provides robust marketplace-wide price estimation.
