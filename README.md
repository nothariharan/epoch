# 🧠 MindfulAI - Technical Documentation

### 🔍 1. Project Overview

**MindfulAI** is a multi-modal prototype designed for the *"AI for Early Mental Health Detection"* hackathon.  
The goal is to analyze **user-generated text and visual data** to detect early indicators of stress and provide **supportive, actionable insights** in a secure and ethical manner.

The application is built as a **two-column Streamlit dashboard**:

- **Text Analysis:** Analyzes a user's journal entry for stress.  
- **Visual Check-in:** Analyzes an uploaded facial image for emotional indicators of stress.

---

### 🧪 2. Model 1: Text-Based Stress Analysis

#### 2.1 Dataset

- **Name:** `Stress.csv` (popular Kaggle dataset)  
- **Source:** Derived from mental health-related subreddits (e.g., `r/stress`)  
- **Structure:**  
  - `text` → user's post  
  - `label` → binary indicator (1 = Stress, 0 = No Stress)

---

#### 2.2 Preprocessing

A custom Python function `clean_text()` is used to prepare the data:

- **Lowercase:** Converts all text to lowercase.  
- **Punctuation/Noise Removal:** Uses `re` to remove all non-alphabetic characters.  
- **Stopword Removal:** Uses `nltk.corpus.stopwords` to remove common English words (e.g., "the", "is", "a").

---

#### 2.3 Model Architecture & Training Pipeline

The model uses a **scikit-learn pipeline** with two main stages:

**1. Vectorization (Feature Extraction):**
- **Technique:** `TfidfVectorizer`
- **Why:** TF-IDF weighs words based on importance, giving rare, meaningful words more influence.  
- **Tuning:** `ngram_range=(1, 2)` captures both single words and word pairs (e.g., `"so overwhelmed"`).

**2. Classification (Model):**
- **Algorithm:** `LogisticRegression`
- **Why:** Fast, interpretable, and provides probability-based scores.

**Training Pipeline Steps:**
1. Load `Stress.csv` into a pandas DataFrame.  
2. Apply `clean_text()` on the text column.  
3. Split data → 80% training, 20% testing.  
4. Fit-transform TF-IDF on training data.  
5. Train Logistic Regression on vectorized text.  
6. Save model and vectorizer as `model.pkl` and `vectorizer.pkl`.

---

#### 2.4 Evaluation & Output

- **Accuracy:** ~75% on unseen test data.  
- **Live Output:** Uses `model.predict_proba()` to generate a **Stress Likelihood Score** (e.g., `62.25%`).  
- **Interpretation:**  
  - 0–40% → Low risk  
  - 41–70% → Moderate risk  
  - 71–100% → High risk

---

### 🤖 3. Model 2: Visual Stress-Indicator Analysis

#### 3.1 Dataset & Training

- **Model:** Pre-trained model via the `deepface` library.  
- **Strategy:** Use a state-of-the-art, pre-trained model instead of training from scratch (hackathon constraint).  
- **Training Data:** DeepFace emotion model trained on academic datasets like **FER2013** (hundreds of thousands of labeled faces).

---

#### 3.2 Preprocessing

- User uploads an image (`.jpg`, `.png`).  
- Streamlit reads it as bytes.  
- `cv2.imdecode()` converts bytes into a NumPy array (image).  
- Internal steps like **face detection**, **cropping**, **alignment**, and **normalization** are automatically handled by `DeepFace.analyze()`.

---

#### 3.3 Model Architecture

- **Type:** Convolutional Neural Network (CNN)  
- **Function:** Recognizes facial features (eyes, mouth, eyebrows) and classifies them into 7 primary emotions:
  - angry  
  - disgust  
  - fear  
  - happy  
  - sad  
  - surprise  
  - neutral

---

#### 3.4 Evaluation & Output

- **Raw Output:** `DeepFace.analyze()` returns probabilities for all 7 emotions  
  Example:
  ```python
  {'fear': 71.26, 'sad': 8.95, 'happy': 5.34, ...}
- **Custom Metric** – Calculated Stress Score

- **Stress Score Formula:**

- `Stress Score = %angry + %disgust + %fear + %sad`

- Combines multiple stress-related emotions to better reflect emotional strain.
Example: 50% sad + 30% angry → 80% stress = High stress

**Risk Mapping:**

- 0–40% → Low

- 41–70% → Moderate

- 71–100% → High

### ⚖️ 4. Ethical Considerations

Ethics were a primary focus in this project.

- **🩺 Not a Diagnostic Tool**

MindfulAI is positioned as a personal assistant — not a medical diagnostic device.

- **🔒 Data Privacy (Most Important Feature)**

All processing happens locally in the user’s browser session.

No data is saved to disk or sent to any external server.

Once the tab is closed, all data is permanently deleted.

- **⚠️ Model Bias Acknowledgement**

- Text Model: May reflect biases from Reddit user demographics.

- Visual Model: Facial recognition models can vary in performance across ethnicities.

- **💬 Responsible Suggestions**

Instead of only displaying a "High" risk warning, the app provides supportive resources:

For "High" risk users (India-specific helplines):

- ☎️ Kiran Helpline: 1800-599-0019

- ☎️ Vandrevala Foundation Helpline: 9999-666-555
