# Machine Learning Evaluation Metrics

## Classification

### Precision

Precision measures the accuracy of positive predictions. It's calculated as:

```
Precision = True Positives / (True Positives + False Positives)
```

Precision answers: "Of all instances predicted as positive, what percentage is actually positive?"

### Recall

Recall (also called Sensitivity or True Positive Rate) measures the ability to find all positive instances. It's calculated as:

```
Recall = True Positives / (True Positives + False Negatives)
```

Recall answers: "Of all actual positive instances, what percentage was correctly identified?"

### F1 Score

F1 score is the harmonic mean of precision and recall, providing a balance between the two:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

F1 is useful when you need to balance precision and recall, especially with imbalanced datasets.

### ROC AUC

Receiver Operating Characteristic Area Under Curve (ROC AUC) measures the model's ability to discriminate between positive and negative classes across various thresholds. It plots True Positive Rate (Recall) against False Positive Rate. A perfect model has ROC AUC = 1, while random guessing gives 0.5.

## P/R AUC

Precision-Recall Area Under Curve (P/R AUC) plots precision against recall at different thresholds. It's particularly useful for imbalanced datasets where ROC AUC might be misleadingly optimistic.

## mAP

Mean Average Precision (mAP) is commonly used in object detection and information retrieval. It calculates the mean of Average Precision (AP) across multiple classes or queries. AP summarizes the precision-recall curve into a single value.

## Log-loss

Logarithmic loss (log-loss) measures the performance of a classification model where the prediction is a probability value between 0 and 1:

```
Log-loss = -1/N * Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

Lower log-loss indicates better performance. It heavily penalizes confident incorrect predictions.

## Rank

## Precision@k

Precision@k measures the proportion of relevant items among the top-k retrieved items. It's calculated as:

```
Precision@k = (Number of relevant items in top-k) / k
```

Precision@k answers: "What percentage of the top-k recommended items are relevant to the user?"

## Recall@k

Recall@k measures the proportion of relevant items that appear in the top-k results. It's calculated as:

```
Recall@k = (Number of relevant items in top-k) / (Total number of relevant items)
```

Recall@k answers: "What percentage of all relevant items are included in the top-k recommendations?" Unlike ranking metrics, it does not consider the quality of the ranking.

## MRR

Mean Reciprocal Rank (MRR) measures the effectiveness of a retrieval system based on the rank of the first relevant result. It's calculated as:

```
MRR = 1/|Q| * Σ(1/rank_i)
```

Where |Q| is the number of queries and rank_i is the position of the first relevant result for the i-th query. MRR focuses on the first relevant item's position.

## nDCG

Normalized Discounted Cumulative Gain (nDCG) measures the ranking quality of search and recommendation systems. It's calculated by normalizing DCG:

```
DCG@k = Σ(rel_i / log₂(i+1))
nDCG@k = DCG@k / IDCG@k
```

Where rel_i is the relevance of the i-th item and IDCG is the ideal DCG. nDCG values range from 0 to 1, with 1 representing perfect ranking.

## Regression

## MSE

Mean Squared Error (MSE) measures the average squared difference between predicted and actual values in regression tasks:

```
MSE = 1/n * Σ(y_i - ŷ_i)²
```

MSE penalizes larger errors more heavily due to the squaring operation.

## MAE

Mean Absolute Error (MAE) measures the average absolute difference between predicted and actual values:

```
MAE = 1/n * Σ|y_i - ŷ_i|
```

MAE is more robust to outliers than MSE and provides a linear penalty for errors.

## Language

## BLEU

Bilingual Evaluation Understudy (BLEU) measures the quality of machine-translated text by comparing it to reference translations. It calculates n-gram precision with a brevity penalty:

```
BLEU = BP * exp(Σ w_n * log p_n)
```

Where BP is the brevity penalty and p_n is the precision for n-grams. BLEU scores range from 0 to 1, with higher values indicating better translations.

## BLEURT

Bilingual Evaluation Understudy with Representations from Transformers (BLEURT) is a learned metric for evaluating machine translation quality. It uses a fine-tuned BERT model to predict human judgments of translation quality, offering more nuanced evaluation than traditional metrics.

## GLUE

General Language Understanding Evaluation (GLUE) is a benchmark for evaluating natural language understanding systems across multiple tasks, including sentiment analysis, paraphrasing, and inference. It provides a standardized way to compare language models' performance.

## ROUGE

Recall-Oriented Understudy for Gisting Evaluation (ROUGE) measures the quality of summaries by comparing them to reference summaries. It focuses on recall of n-grams, word sequences, and word pairs:

```
ROUGE-N = Σ(count_match(n-gram)) / Σ(count(n-gram))
```

ROUGE is commonly used in text summarization evaluation.

## Ads

## CPE

Cost Per Engagement (CPE) is an advertising metric that measures the cost effectiveness of ad campaigns:

```
CPE = Total Cost / Number of Engagements
```

Lower CPE indicates more cost-effective campaigns. It's particularly important for optimizing ad recommendation systems.

## Latency

Latency measures the time delay between a request and the corresponding response in a machine learning system. It's critical for real-time applications and user experience:

```
Latency = Response Time - Request Time
```

Latency is typically measured in milliseconds and is a key operational metric for deployed ML systems.

## Computational Cost

Computational cost measures the resources required to train or run a machine learning model, including:

```
Computational Cost = f(CPU/GPU time, Memory usage, Energy consumption)
```

This metric is particularly important for on-device applications where resources are limited and battery life is a concern.This metric is particularly important for on-device applications where resources are limited and battery life is a concern.

# Tradeoff between metrics

When evaluating machine learning models, optimizing for one metric often comes at the expense of others. Understanding these tradeoffs is crucial for making informed decisions about model selection and optimization.

## Precision vs. Recall
One of the most common tradeoffs is between precision and recall. Increasing the classification threshold typically improves precision but reduces recall, and vice versa. This tradeoff is visualized in the precision-recall curve.

## Accuracy vs. Fairness
Models optimized solely for accuracy may perform poorly on fairness metrics. For example, a facial recognition system might achieve high overall accuracy while performing significantly worse on certain demographic groups.

## Performance vs. Interpretability
Complex models like deep neural networks often achieve higher performance metrics but are less interpretable than simpler models like decision trees or linear regression.

## Offline vs. Online Metrics
Models that perform well on offline metrics (like AUC or F1) may not necessarily improve online business metrics (like CTR or revenue). This discrepancy can occur due to differences between training data distribution and real-world data.

## Latency vs. Accuracy
More accurate models often require more computation, leading to higher latency. In real-time applications, a slightly less accurate model with faster inference time might be preferable.

## Generalization vs. Memorization
Models with high training accuracy might be memorizing the training data rather than learning generalizable patterns, leading to poor performance on new data.

## Short-term vs. Long-term Metrics
Optimizing for short-term engagement metrics (like CTR) might harm long-term user satisfaction and retention. It's important to balance immediate performance with sustainable growth.

## Balancing Tradeoffs
To effectively balance these tradeoffs:
1. Define clear business objectives
2. Use multiple complementary metrics
3. Consider the specific constraints of your application
4. Monitor both offline and online performance
5. Regularly reassess metric priorities as business needs evolve

# Online Metrics

Online metrics measure the performance of machine learning models in real-world production environments with actual users. These metrics are crucial for understanding the business impact of your models.

## CTR
Click-Through Rate (CTR) measures the ratio of users who click on a specific link to the number of total users who view a page, email, or advertisement. It's calculated as:

```
CTR = (Number of Clicks / Number of Impressions) * 100%
```

CTR is widely used in search, advertising, and recommendation systems to evaluate user engagement.

## Task/Session Success Rate
Task or session success rate measures the percentage of user sessions that successfully accomplish a defined goal or task. It's calculated as:

```
Success Rate = (Number of Successful Sessions / Total Number of Sessions) * 100%
```

This metric directly reflects how well your model helps users achieve their objectives.

## Task/Session Duration
Task or session duration measures the total time users spend engaging with your product during a session. Longer durations often indicate higher user engagement, particularly for content consumption platforms like video streaming services.

```
Average Session Duration = Total Session Time / Number of Sessions
```

However, interpretation depends on the context—shorter durations might be preferred for task-completion scenarios.

## Engagement Rate
Engagement rate measures how users interact with content beyond simple views, including actions like likes, comments, shares, or saves. It's calculated as:

```
Engagement Rate = (Total Engagements / Total Impressions) * 100%
```

Higher engagement rates typically indicate more compelling content or recommendations.

## Conversion Rate
Conversion rate measures the percentage of users who take a desired action (purchase, signup, download, etc.). It's calculated as:

```
Conversion Rate = (Number of Conversions / Total Number of Visitors) * 100%
```

This is one of the most direct metrics for measuring business impact, especially in e-commerce and lead generation.

## Revenue Lift
Revenue lift measures the incremental revenue generated by a model compared to a baseline or control group. It's calculated as:

```
Revenue Lift = ((Revenue with Model - Revenue without Model) / Revenue without Model) * 100%
```

This metric directly ties machine learning performance to business outcomes and ROI.

## Reciprocal Rank of First Click
Reciprocal Rank of First Click measures how quickly users find what they're looking for, based on the position of their first click. It's calculated as:

```
Reciprocal Rank = 1 / Position of First Click
```

A higher reciprocal rank indicates that users are finding relevant results earlier in the list, suggesting better ranking quality.

## Counter Metrics
Counter metrics track negative user feedback that directly indicates dissatisfaction with the model's output. Examples include:

- Hide rate: Percentage of users who hide or dismiss recommended content
- Report rate: Percentage of users who report content as inappropriate or irrelevant
- Bounce rate: Percentage of users who leave immediately after viewing a single page

These metrics serve as important guardrails and can highlight potential issues with your model that might not be captured by positive engagement metrics.
