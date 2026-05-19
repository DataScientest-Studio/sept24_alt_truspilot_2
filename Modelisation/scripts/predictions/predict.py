import joblib

MODEL_PATH = "models/trustpilot_logistic_tfidf.joblib"


def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def predict_text(text: str) -> dict:
    model = load_model()

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    return {
        "text": text,
        "prediction": int(prediction),
        "probability_negative": float(probabilities[0]),
        "probability_positive": float(probabilities[1]),
    }


if __name__ == "__main__":
    example_text = "The product is amazing, I am very satisfied."
    result = predict_text(example_text)
    print(result)