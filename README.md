# Sentiment Analyzer using HuggingFace Transformers

## Overview

This project is a sentiment analysis tool that classifies text as Positive, Negative, or Neutral using a pre-trained transformer model from HuggingFace.

## Model Used

I used `cardiffnlp/twitter-roberta-base-sentiment-latest`, a 3-label sentiment model. I chose this model because the assignment requires Positive, Negative, and Neutral labels.

## How to Run

1. Install dependencies:

```bash
pip install transformers torch pandas scikit-learn
```

2. Run the script:

```bash
python main.py
```

3. Enter a sentence in the terminal to get its sentiment prediction and confidence score.

## Files Included

* `main.py`
* `results.csv`
* `README.md`

## Results

The model was tested on 50 hand-labeled examples and compared against manual labels.

## Observations

The model performed well on clearly positive and clearly negative examples. Most incorrect predictions happened on neutral or slightly ambiguous sentences such as “It’s decent” or “Average experience,” where sentiment can be interpreted in different ways.

## Conclusion

This project shows how a pre-trained transformer model can be used for sentiment analysis without training a model from scratch. It also shows that neutral sentiment is more difficult to classify than strong positive or negative sentiment.