# A/B Testing in Machine Learning Systems

A/B testing (also called split testing or bucket testing) is a controlled experiment methodology used to compare two or more versions of a system to determine which performs better according to predefined metrics. In ML systems, A/B testing is crucial for validating model improvements, feature changes, or new algorithms in real-world environments.

## Core Concepts

### Experimental Design

1. **Test and Control Groups**
   - **Control Group (A)**: Users exposed to the current system/model (baseline)
   - **Test Group (B)**: Users exposed to the new version or variant
   - **Multiple Variants**: Can extend to A/B/n testing with multiple test variants

2. **Randomization**
   - Users must be randomly assigned to groups
   - Randomization must be consistent for the same user (sticky assignment)
   - Often implemented using hashing of user IDs with modulo operation

3. **Sample Size Determination**
   - Based on:
     - Minimum Detectable Effect (MDE)
     - Statistical significance level (typically α = 0.05)
     - Statistical power (typically 1-β = 0.8)
   - Larger sample sizes needed for detecting smaller effects

4. **Traffic Allocation**
   - Standard allocation: 50/50 split for two variants
   - Uneven splits (e.g., 90/10) for risky changes or initial testing
   - Ramped rollouts: gradually increasing test group size

## Statistical Framework

1. **Hypothesis Testing**
   - **Null hypothesis (H₀)**: No difference between variants
   - **Alternative hypothesis (H₁)**: There is a difference between variants
   - **p-value**: Probability of observing the results (or more extreme) if H₀ were true
   - **Significance level (α)**: Threshold for rejecting H₀ (typically 0.05)

2. **Common Statistical Tests**
   - **t-test**: For continuous metrics (e.g., session duration, revenue per user)
   - **z-test**: For proportions (e.g., click-through rate, conversion rate)
   - **Mann-Whitney U test**: Non-parametric alternative when normality cannot be assumed
   - **CUPED** (Controlled-experiment Using Pre-Experiment Data): Variance reduction technique

3. **Multiple Testing Problem**
   - Risk of false positives increases with number of metrics
   - Corrections:
     - Bonferroni correction
     - False Discovery Rate (FDR) control
     - Closed testing procedures

## Implementation Considerations

1. **Technical Implementation**
   - **Experiment Assignment Service**: Ensures consistent user assignment
   - **Feature Flagging**: Controls feature exposure at runtime
   - **Experiment Configuration**: Defines parameters, traffic allocation, metrics
   - **Logging Infrastructure**: Captures user interactions and relevant metrics

2. **Duration Planning**
   - Sufficient duration to account for:
     - Weekly seasonality (typically minimum 1-2 weeks)
     - Enough samples for statistical significance
     - Novelty effects or learning curves
     - Long-term impacts (some experiments need extended observation)

3. **Segmentation Analysis**
   - Breaking down results by user segments:
     - New vs. returning users
     - Mobile vs. desktop
     - Geographic regions
     - User activity levels

4. **Guardrail Metrics**
   - Metrics monitored to ensure experiment isn't causing harm:
     - Site performance (latency, error rates)
     - Core business metrics (revenue, retention)
     - User satisfaction metrics

## Common Challenges and Solutions

1. **Sample Ratio Mismatch (SRM)**
   - Problem: Actual traffic split differs from intended split
   - Causes: Data loss, tracking issues, implementation bugs
   - Detection: Chi-squared test on assignment counts
   - Solution: Fix technical issues causing uneven assignment or data capture

2. **Network Effects and Interference**
   - Problem: Users in different groups influencing each other
   - Examples: Social features, marketplace dynamics
   - Solutions:
     - Cluster-based randomization
     - Switchback experiments
     - Synthetic controls

3. **Simpson's Paradox**
   - Problem: Trend appears in groups but disappears or reverses when groups combined
   - Solution: Stratified analysis and controlling for confounding variables

4. **Novelty and Primacy Effects**
   - Problem: Temporary changes in behavior due to newness or change itself
   - Solution: Run experiments long enough for behavior to stabilize

## Advanced A/B Testing Techniques

1. **Multi-Armed Bandits (MAB)**
   - Dynamically adjusts traffic allocation to favor better-performing variants
   - Balances exploration (learning) and exploitation (optimizing)
   - Variations:
     - Epsilon-greedy
     - Thompson sampling
     - Upper Confidence Bound (UCB)

2. **Sequential Testing**
   - Evaluates results continuously rather than just at experiment end
   - Allows for earlier stopping when significant results detected
   - Requires statistical corrections for multiple looks

3. **Interleaving Experiments**
   - Particularly useful for ranking systems
   - Mixes results from different algorithms and observes user preference
   - More sensitive than traditional A/B tests for detecting ranking improvements

4. **Quasi-Experiments**
   - Used when randomization is not possible
   - Techniques:
     - Difference-in-differences
     - Regression discontinuity
     - Synthetic control methods

## ML-Specific Considerations

1. **Online Evaluation Metrics**
   - Click-through rate (CTR)
   - Conversion rate
   - User engagement (time spent, interactions)
   - Revenue metrics (ARPU, LTV)
   - Satisfaction metrics (ratings, NPS)

2. **Offline vs. Online Evaluation Discrepancies**
   - Causes:
     - Distribution shifts between offline and online data
     - Missing contextual factors in offline evaluation
     - Feedback loops in online environments
   - Solution: Establish correlation between offline and online metrics

3. **Ramp-up Strategies for ML Models**
   - Shadow mode testing (collecting data without affecting users)
   - A/A testing to validate experiment infrastructure
   - Gradual traffic allocation increasing over time
   - Holdback experiments (comparing new model to no model)

4. **Long-term Impact Assessment**
   - Longer experiments to capture delayed effects
   - Longitudinal analysis of user cohorts
   - Holdout groups maintained for extended periods

## Best Practices

1. **Experiment Documentation**
   - Clear hypothesis statement
   - Pre-registered primary and secondary metrics
   - Expected effect sizes and directions
   - Detailed experimental design

2. **Incremental Testing**
   - Test one change at a time when possible
   - Use factorial designs for testing multiple factors
   - Build on successful experiments incrementally

3. **Result Interpretation**
   - Consider practical significance, not just statistical significance
   - Look for heterogeneous treatment effects across segments
   - Correlate results with qualitative user feedback
   - Investigate unexpected results thoroughly

4. **Organizational Setup**
   - Experimentation platform that standardizes process
   - Clear decision criteria for shipping changes
   - Culture that respects experimental results
   - Knowledge sharing of experiment outcomes
