# Results Summary — Approach 1 vs Approach 2

# Marketplace Price Prediction Challenge

This document compares the two modelling approaches explored during the project:

1. **Approach 1 — Global Marketplace Model (XGBoost, no IDs)**
2. **Approach 2 — Entity-Conditioned Model (CatBoost with IDs)**

The goal is to understand:

* how the two approaches differ
* why their performances differ
* why the simpler global model ultimately performed better

---

# Problem Context

The task is to predict marketplace prices using:

* historical marketplace behaviour
* lagged pricing signals
* historical aggregates
* sparse future observations

Inference data contains:

* many rows with only IDs
* a small daily anchor set with true prices

---

# Approach 1 — Global Marketplace Model

## Core Idea

Train a single global model using:

* marketplace behavioural features
* historical price statistics
* lag features
* trend features
* discount features

Importantly:

```
No raw IDs were used directly.
```

The model generalises using:

* pricing behaviour
* historical dynamics
* marketplace structure

rather than memorising entities.

---

# Model

## Algorithm

```
XGBoost Regressor
```

## IDs Used?

```
No
```

---

# Validation Result

| Metric | Result       |
| ------ | ------------ |
| RMSE   | 1,170,620.94 |
| MAE    | 237,091.23   |
| MAPE   | 0.32%        |

---

# Top Features

| Feature                       | Importance |
| ----------------------------- | ---------- |
| item_price_last_scrape_log1p  | 0.447660   |
| item_price_lag_2_log1p        | 0.249527   |
| model_price_last_scrape_log1p | 0.104069   |
| item_has_price_history        | 0.081859   |
| item_has_history              | 0.046509   |

---

# Key Behaviour

The model relies overwhelmingly on:

* historical prices
* lag consistency
* marketplace price persistence

This indicates:

```
marketplace prices are extremely sticky
```

and:

* recent historical price is already a very strong predictor.

---

# Approach 2 — Entity-Conditioned Model

## Core Idea

Train a more personalised model using:

* raw IDs
* historical features
* lag features

The hypothesis was:

```
different shops/products have unique pricing behaviour
```

and therefore:

* entity conditioning should improve accuracy.

---

# Model

## Algorithm

```
CatBoost Regressor
```

## IDs Used?

```
Yes
```

Direct categorical inputs:

* `shopId`
* `itemId`
* `modelId`

---

# Validation Result

| Metric | Result       |
| ------ | ------------ |
| RMSE   | 2,835,104.63 |
| MAE    | 502,094.57   |
| MAPE   | 0.64%        |

---

# Top Features

| Feature                       | Importance |
| ----------------------------- | ---------- |
| item_price_last_scrape_log1p  | 38.609495  |
| item_price_lag_2_log1p        | 10.711625  |
| model_price_max_hist_log1p    | 7.784619   |
| model_price_last_scrape_log1p | 6.690715   |
| model_avg_price_hist_log1p    | 5.325866   |
| shopId                        | 2.586797   |
| itemId                        | 0.976108   |

---

# Direct Comparison

| Aspect              | Approach 1 | Approach 2 |
| ------------------- | ---------- | ---------- |
| Model               | XGBoost    | CatBoost   |
| Uses IDs            | No         | Yes        |
| Entity Conditioning | Indirect   | Direct     |
| MAPE                | **0.32%**  | 0.64%      |
| RMSE                | **1.17M**  | 2.84M      |
| Generalisation      | Strong     | Weaker     |
| Complexity          | Lower      | Higher     |

---

# Why Approach 1 Performed Better

# 1. Marketplace Prices Are Extremely Stable

The strongest signals in both approaches were:

* previous item price
* lagged prices
* historical averages

This means:

```
most products simply retain their previous prices
```

As a result:

* complex entity conditioning adds limited extra information.

---

# 2. Historical Behaviour Already Encodes Entity Information

Although Approach 1 does not directly use IDs, its features already capture:

* item pricing history
* model-level behaviour
* volatility
* trend
* historical range

Effectively:

```
entity behaviour was already compressed into engineered features
```

This allowed:

* strong generalisation
* simpler modelling
* less overfitting

---

# 3. Direct IDs Increased Model Complexity

Approach 2 introduced:

* thousands of item IDs
* tens of thousands of model IDs

Even though CatBoost handles categorical variables well, the model now had to learn:

* sparse entity relationships
* fragmented product histories
* high-cardinality category interactions

This increased:

* model complexity
* variance
* overfitting risk

---

# 4. Sparse Entity Histories Hurt Generalisation

Many products:

* appear infrequently
* have limited history
* have missing lag structures

Approach 2 became more sensitive to:

* sparse entity coverage
* unseen entity combinations
* fragmented marketplace states

Whereas Approach 1:

* relied more on stable marketplace-wide dynamics.

---

# 5. The Problem Is Dominated By Price Persistence

Both models show the same dominant features:

| Feature             | Observation |
| ------------------- | ----------- |
| last observed price | dominant    |
| lag-2 price         | dominant    |
| historical averages | strong      |

This indicates the task behaves more like:

```
historical price continuation
```

than:

```
complex personalised pricing prediction
```

As a result:

* simpler models generalise better.

---

# Important Observation

Even in Approach 2:

* IDs were NOT among the top predictive features.

For example:

| Feature | Importance |
| ------- | ---------- |
| shopId  | 2.59       |
| itemId  | 0.98       |

compared to:

| Feature             | Importance |
| ------------------- | ---------- |
| previous item price | 38.61      |
| lag-2 item price    | 10.71      |

This strongly suggests:

```text id="cwmgph"
historical pricing behaviour matters far more than raw identity
```

---

# Final Conclusion

## Approach 1 Was Superior For This Dataset

The global marketplace XGBoost model achieved:

* lower RMSE
* lower MAE
* lower MAPE
* better generalisation

despite being:

* simpler
* faster
* lower complexity

---

# Main Reason

The dataset exhibits:

```
extremely strong short-term price persistence
```

Therefore:

* historical lag features dominate predictive power
* direct entity modelling adds limited incremental value

---

# Key Takeaway

For this marketplace problem:

```
well-engineered historical behavioural features outperform direct entity memorisation
```

The best-performing strategy was:

* marketplace-wide modelling
* strong lag/history engineering
* anchor-based calibration
* robust generalisation

rather than:

* aggressive entity conditioning.
