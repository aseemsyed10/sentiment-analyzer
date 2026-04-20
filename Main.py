import pandas as pd
import re
from transformers import pipeline
from sklearn.metrics import accuracy_score

clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def map_label(label):
    label = str(label).strip().lower()

    if label == "negative":
        return "Negative"
    elif label == "neutral":
        return "Neutral"
    elif label == "positive":
        return "Positive"
    else:
        return "Unknown"

def analyze_sentiment(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".!?,")

    result = clf(text)[0]
    raw_label = result["label"]
    label = map_label(raw_label)
    score = result["score"]
    return label, score

data = [
    ("I loved this movie, it was fantastic!", "Positive"),
    ("This was the worst experience ever.", "Negative"),
    ("The product is okay, nothing special.", "Neutral"),
    ("Absolutely amazing performance!", "Positive"),
    ("I hate this so much.", "Negative"),
    ("It’s fine, not too bad.", "Neutral"),
    ("Best purchase I’ve made!", "Positive"),
    ("Terrible quality, very disappointed.", "Negative"),
    ("It works as expected.", "Neutral"),
    ("I’m extremely happy with this.", "Positive"),
    ("Not worth the money.", "Negative"),
    ("Average experience.", "Neutral"),
    ("Highly recommend this!", "Positive"),
    ("Worst product ever.", "Negative"),
    ("It’s decent.", "Neutral"),
    ("I really enjoyed using this.", "Positive"),
    ("Very bad service.", "Negative"),
    ("Nothing impressive.", "Neutral"),
    ("Excellent quality!", "Positive"),
    ("I regret buying this.", "Negative"),
    ("I’m satisfied with the result.", "Positive"),
    ("This is awful and frustrating.", "Negative"),
    ("It is just normal.", "Neutral"),
    ("Superb and wonderful!", "Positive"),
    ("This is disappointing.", "Negative"),
    ("Not bad, just average.", "Neutral"),
    ("I’m very pleased with this item.", "Positive"),
    ("Completely useless product.", "Negative"),
    ("It was neither good nor bad.", "Neutral"),
    ("Fantastic work by the team.", "Positive"),
    ("I would never buy this again.", "Negative"),
    ("The experience was okay overall.", "Neutral"),
    ("This made my day!", "Positive"),
    ("Extremely poor performance.", "Negative"),
    ("It’s acceptable.", "Neutral"),
    ("I’m impressed by the quality.", "Positive"),
    ("The item broke after one use.", "Negative"),
    ("Nothing to complain about.", "Neutral"),
    ("One of the best things I’ve used.", "Positive"),
    ("Customer support was horrible.", "Negative"),
    ("It was just fine.", "Neutral"),
    ("I truly love this product.", "Positive"),
    ("This is a complete disaster.", "Negative"),
    ("It’s manageable, I guess.", "Neutral"),
    ("Brilliant experience from start to finish.", "Positive"),
    ("I’m upset with the purchase.", "Negative"),
    ("This seems pretty standard.", "Neutral"),
    ("Very reliable and easy to use.", "Positive"),
    ("Waste of money.", "Negative"),
    ("It’s okay for the price.", "Neutral")
]

results = []

for text, true_label in data:
    predicted_label, score = analyze_sentiment(text)
    results.append({
        "text": text,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "score": score
    })

df = pd.DataFrame(results)

df["true_label"] = df["true_label"].astype(str).str.strip()
df["predicted_label"] = df["predicted_label"].astype(str).str.strip()

accuracy = accuracy_score(df["true_label"], df["predicted_label"])
print("Accuracy:", round(accuracy * 100, 2), "%")

correct = df[df["true_label"] == df["predicted_label"]]
incorrect = df[df["true_label"] != df["predicted_label"]]

print("Correct predictions:", len(correct))
print("Incorrect predictions:", len(incorrect))

df.to_csv("results.csv", index=False)
print("results.csv saved successfully.")

def sentiment_cli():
    print("Sentiment Analyzer")
    print("Type 'quit' to stop.\n")

    while True:
        user_input = input("Enter text: ")

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if len(user_input.split()) < 2:
            print("⚠️ Please enter a more complete sentence.\n")
            continue

        label, score = analyze_sentiment(user_input)

        print(f"Sentiment: {label}")
        print(f"Confidence Score: {score:.4f}\n")

sentiment_cli()