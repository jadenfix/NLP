# NLP
N-Gram Language Model
A statistical language modeling implementation supporting Unigram, Bigram, and Trigram architectures with configurable smoothing techniques.

Core Features
N-Gram Architectures:

Unigram (1-gram), Bigram (2-gram), and Trigram (3-gram) implementations

Backoff mechanism through nested model dependencies (Trigram → Bigram → Unigram)

Smoothing Methods:

Add-k Smoothing: Applied to Unigram models via l1 hyperparameter

Linear Interpolation: Weighted combination of different n-gram orders (l1, l2, l3 coefficients)

Data Handling:

Automatic <UNK> token replacement for low-frequency words (configurable frequency threshold)

<START>/<STOP> token injection for sequence boundary modeling

Evaluation:

Perplexity calculation with log-probability aggregation

Zero-probability mitigation via 1e-10 floor value

Model Workflow
Corpus Preprocessing:

Whitespace tokenization with boundary markers

Vocabulary frequency analysis and rare word substitution

Probability Estimation:

Maximum Likelihood Estimation (MLE) for raw n-gram counts

Smoothing-enabled conditional probability calculations:

P(wₜ|wₜ₋₁) = λ₂Pₘₗₑ(wₜ|wₜ₋₁) + λ₁Pₘₗₑ(wₜ)

Cross-Validation:

Train/dev/test split evaluation

Hyperparameter tuning experiments:

Training data subsampling (50% ablation study)

Frequency constraint variations (3 vs 5 occurrences)

Technical Implementation
Class Hierarchy:

Base unigram → Bigram (with unigram dependency) → Trigram (with bigram dependency)

Inherited perplexity calculation with n-gram order-specific probability chaining

Efficiency Considerations:

Frequency tables implemented via collections.Counter

Probability caching during training phase

Experimental Results
Perplexity Trends:

Expected degradation from Unigram → Bigram → Trigram due to data sparsity

Smoothing improves generalization (lower dev/test perplexity vs train)

Optimal interpolation weights found empirically through grid search

Robustness Tests:

Performance evaluation with reduced training data (50% subsample)

Impact analysis of UNK thresholding (frequency constraint = 3 vs 5)
