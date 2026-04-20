# Sentiment Analyzer using HuggingFace Transformers

## 📌 Overview

This project implements a sentiment analysis tool that classifies text into **Positive, Negative, or Neutral** categories using a pre-trained transformer model from HuggingFace. The system takes user input, processes it through the model, and returns both the predicted sentiment label and a confidence score.

---

## 🤖 Model Used

I used the **cardiffnlp/twitter-roberta-base-sentiment-latest** model, which is based on RoBERTa and supports **three sentiment classes**:

* Negative
* Neutral
* Positive

This model was selected because the default HuggingFace sentiment model is binary (Positive/Negative), while the assignment requires three classes including Neutral.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install transformers torch pandas scikit-learn
```

### 2. Run the script

```bash
python main.py
```

### 3. Use the CLI

* Enter any sentence to analyze its sentiment
* Type `quit` to exit

---

## 📂 Files Included

* `main.py` → Main Python script
* `results.csv` → 50 test examples with predictions
* `README.md` → Project documentation

---

## 📊 Results

The model was tested on **50 hand-labeled examples**.

**Accuracy: ~85%**  

Each entry in `results.csv` contains:

* Input text
* True label
* Predicted label
* Confidence score

---

## ❌ Error Analysis

Some incorrect predictions occurred on neutral or ambiguous sentences:

* **"Average experience."** → Predicted: Negative
* **"It's decent."** → Predicted: Positive
* **"Nothing impressive."** → Predicted: Negative

These errors occur because such phrases can be interpreted differently depending on context, making them difficult to classify precisely.

---
## ⚠️ Challenges Faced

During the development of this project, several challenges were encountered:

- **Model Selection Issue:**  
  The default HuggingFace sentiment model only supports binary classification (Positive/Negative). Since the assignment required three classes (Positive, Negative, Neutral), a different model had to be researched and selected.

- **Label Mapping Confusion:**  
  Initially, the model outputs were not mapped correctly, leading to incorrect or `None` predictions. This required debugging and adjusting the label mapping function.

- **Handling Neutral Sentiment:**  
  Neutral sentences were difficult to classify accurately because they often contain subtle or ambiguous language. The model sometimes interpreted neutral text as slightly positive or negative.

- **Input Sensitivity:**  
  Small changes in input, such as punctuation (e.g., "It's decent" vs "It's decent."), resulted in different confidence scores, showing that the model is sensitive to exact text formatting.

- **User Input Validation:**  
  Short or incomplete inputs (e.g., single words or numbers) led to unreliable predictions, requiring additional checks in the CLI to ensure meaningful input.

---
## 🧠 Observations

* The model performs very well on **clearly positive and clearly negative text**.
* It struggles more with **neutral or ambiguous sentences**, where sentiment is subtle.
* Confidence scores are higher for strong sentiment and lower for uncertain inputs.
* Small changes in wording or punctuation can slightly affect prediction confidence.

---

## ⚠️ Limitations

* Neutral sentiment is harder to classify than positive or negative.
* The model may misinterpret short or incomplete text.
* Predictions depend heavily on wording and context.

---

## 📌 Conclusion

This project demonstrates how transformer-based NLP models can be used for real-world sentiment analysis tasks. While the model achieves high accuracy on clear cases, handling neutral sentiment remains a challenge, highlighting the complexity of natural language understanding.

---
