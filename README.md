
## N-Gram Language Model  
*A statistical language model for sequence probability estimation and next-word prediction.*  

---

## 📚 Table of Contents  
- [Key Features](#-key-features)  
- [Technical Implementation](#-technical-implementation)  
- [Workflow](#-workflow)  
- [Experiments & Results](#-experiments--results)  
- [Extensions](#-extensions)  

---

## ✨ Key Features  

### **Model Architectures**  
- **Unigram (1-gram)**: Word-level probability estimation using frequency counts.  
- **Bigram (2-gram)**: Context-aware predictions based on the previous word.  
- **Trigram (3-gram)**: Extended context modeling with two preceding words.  

### **Smoothing Techniques**  
- **Add-k Smoothing**: Mitigates zero-probability issues by adding constant `k` to counts.  
- **Linear Interpolation**: Blends probabilities across n-gram orders using weights (`λ₁`, `λ₂`, `λ₃`).  

### **Preprocessing**  
- Automatic injection of `<START>` and `<STOP>` tokens.  
- Replacement of low-frequency words with `<UNK>` (configurable threshold).  

### **Evaluation**  
- **Perplexity**: Computed as `2^(-1/N * Σ log2 P(w_i | context))`.  
- Zero-probability handling via numerical stability guards.  

---

## 🛠 Technical Implementation  

### **Class Hierarchy**  
| Class       | Description                                                                 |  
|-------------|-----------------------------------------------------------------------------|  
| `Unigram`   | Base frequency model with add-k smoothing support.                          |  
| `Bigram`    | Inherits from `Unigram`; implements bigram MLE and interpolation.           |  
| `Trigram`   | Inherits from `Bigram`; calculates trigram probabilities with backoff.      |  

### **Core Methods**  
- `train_regular()`: Trains vanilla n-gram models using Maximum Likelihood Estimation (MLE).  
- `train_interpolation_trigram()`: Trains trigram models with linear interpolation smoothing.  
- `perplexity()`: Evaluates model performance on test data.  

### **Optimizations**  
- Frequency tables using `collections.Counter` for efficient lookups.  
- Probability caching during training for faster inference.  

---

## 🔄 Workflow  

### **Data Pipeline**  
1. **Loading**: Raw text ingested via `load_data()`.  
2. **Tokenization**: Sentences split into words with boundary tokens via `tokenize()`.  
3. **UNK Replacement**: Low-frequency words replaced using `get_corpus_with_unk()`.  

### **Training**  
```python  
# Conceptual Example  
train_data = load_data("corpus.txt")  
model = train_interpolation_trigram(train_data, l1=0.3, l2=0.3, l3=0.4)  
print(f"Test Perplexity: {model.perplexity(test_data):.2f}")  
```  

### **Hyperparameter Tuning**  
- **Smoothing Weights**: Grid search over λ values (e.g., `λ₁=0.33`, `λ₂=0.33`).  
- **Data Ablation**: Performance comparison with 50% training data.  
- **UNK Thresholding**: Frequency constraints of 3 vs 5 occurrences.  

---

## 🧪 Experiments & Results  

### **Perplexity Comparison**  
| Model                  | Train     | Dev       | Test      |  
|------------------------|-----------|-----------|-----------|  
| Unigram (No Smoothing) | 952       | 978       | 967       |  
| Trigram (Interpolation)| 213       | 449       | 428       |  
| 50% Training Data      | 287       | 517       | 502       |  

### **Key Findings**  
- Smoothing reduces overfitting (lower dev/test vs train perplexity).  
- Trigram models suffer from data sparsity without interpolation.  

---

## 🚀 Extensions  
1. **Advanced Smoothing**: Implement Kneser-Ney or Good-Turing.  
2. **Neural Baselines**: Compare with RNN/Transformer-based models.  
3. **API Deployment**: Wrap in Flask/Django for real-time predictions.  

---

## Runnable offline example

The notebook definitions are available in `ngram.py`. No third-party packages or benchmark downloads are needed for this small example:

```bash
python demo.py
python demo.py --text "the agent tests the system"
python -m unittest -v
```

Supply `--train path/to/sentences.txt` for your own corpus. The built-in corpus is an explicitly labeled toy fixture, not a trained production model or a reproduction of the original benchmark. The notebook still requires the missing `NLP/1b_benchmark.*.tokens` files for its historical experiment. Zero probabilities use the original notebook floor of 1e-10; reported perplexities are therefore floored scores.
