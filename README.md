# 🛍️ Smart Product Price Prediction using Multimodal AI & Ensemble Learning

## 📌 Overview

This project was developed as part of the **Amazon ML Challenge 2025**, where the goal is to **predict product prices** using textual catalog data and product metadata.

The solution combines:

- Transformer-based text embeddings  
- Engineered structured features  
- Hybrid loss optimization for SMAPE  
- K-Fold training strategy  
- Multi-model ensemble averaging  

The final pipeline is designed to improve prediction robustness and generalization across diverse product categories.

---

## 🎯 Problem Statement

Given product catalog content containing:

- Product title  
- Description  
- Quantity and specifications  

The task is to:

👉 Predict accurate product price  
👉 Optimize predictions using **SMAPE (Symmetric Mean Absolute Percentage Error)**  

---

## 🧠 Approach

The solution follows a **hybrid multimodal modeling pipeline** combining deep learning and feature engineering.

---

# ⚙️ Project Pipeline

---

## 1️⃣ Data Preprocessing & Feature Engineering

A custom PyTorch dataset (`ProductDataset`) was implemented to handle:

### ✅ Text Processing
- Cleaning catalog text
- Tokenization using transformer tokenizer
- Padding and truncation for consistent sequence length

---

### ✅ Structured Feature Extraction

From product descriptions, additional numerical signals were extracted:

- IPQ (Item Packaging Quantity)
- Maximum number presence
- Numeric value count
- Text length
- Word count
- Premium keyword detection

These features are log-transformed for numerical stability.

---

## 2️⃣ Model Architecture

### 🔥 Transformer Backbone

The model uses **BGE (BAAI General Embedding) transformer encoder** to generate contextual text embeddings.

Key Optimization:

- Embedding layers frozen
- Partial fine-tuning of deeper layers
- Reduces overfitting and improves efficiency

---

### 🔥 Attention Pooling

Instead of using only CLS token, attention pooling was applied to:

- Capture important token-level signals
- Improve semantic representation

---

### 🔥 Feature Fusion

Text embeddings were concatenated with engineered structured features using:

- Feature projection layer
- Fully connected regression head
- Residual learning structure

---

## 3️⃣ Loss Function Optimization

A custom hybrid loss was designed:

### ✅ Huber Loss
- Robust against outliers
- Stabilizes training

### ✅ SMAPE Loss
- Direct optimization of competition metric
- Improves leaderboard performance

Final Loss: Hybrid Loss = Huber + SMAPE


---

## 4️⃣ Training Strategy

Two training strategies were implemented:

---

### 🔹 K-Fold Cross Validation
- Improves generalization
- Produces multiple trained models
- Reduces variance

---

### 🔹 Single Train/Validation Split
- Faster experimentation
- Used for rapid iteration

---

### Additional Training Techniques:

- Gradient Accumulation
- Cosine Learning Rate Scheduler
- Warmup Strategy
- Early Stopping
- Gradient Clipping

---

## 5️⃣ Inference & Model Ensembling

During prediction:

- Multiple trained fold models generate predictions
- Predictions are averaged for final output
- Negative predictions are clipped to ensure validity

---

## 6️⃣ External Model Ensemble

Beyond transformer models, predictions from multiple embedding models were combined:

- Stella embedding model
- BGE variants
- Additional tuned submissions

Final ensemble used: Simple Average Across Model Predictions


This improved prediction stability and reduced variance.

---

# 📊 Evaluation Metric

### SMAPE (Symmetric Mean Absolute Percentage Error)

SMAPE = 200 × |Prediction − Actual| / (|Prediction| + |Actual|)


Lower SMAPE indicates better performance.

---

# 🏆 Results

| Model | Validation SMAPE |
|----------|----------------|
| Stella Embedding | ~42.47% |
| BGE Variants | ~46–49% |
| Ensemble Output | Improved stability |

---

# 🛠️ Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas / NumPy
- Attention Mechanisms
- Deep Learning Optimization Techniques

---

## 💡 Key Learnings

- Combining structured and unstructured features improves prediction accuracy.

- Hybrid loss functions help align training objectives with evaluation metrics.

- Model ensembling significantly reduces prediction variance.

- Partial transformer fine-tuning balances performance and efficiency.

## 🔮 Future Improvements

- Incorporating image features for multimodal learning

- Advanced ensemble stacking methods

- Feature selection optimization

- Larger transformer backbone experimentation

---

## 👨‍💻 Author

- Ritesh Raj
- B.Tech Smart Manufacturing | AI/ML Enthusiast
- IIITDM Jabalpur

# ⭐ If You Found This Useful
Please consider starring the repository.