import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter


# ────────────────────────── Veri Yükleme ──────────────────────────
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df = df[["Packet Size", "Local Label"]]
    df["Label"] = df["Local Label"].map({"Normal": 0, "Anomalous": 1})
    X = df[["Packet Size"]]
    y = df["Label"]
    return X, y


# ───────────────────────── Model Eğitimi ─────────────────────────
def train_model(csv_file, model_path="model.pkl"):
    X, y = load_data(csv_file)
    counts = Counter(y)
    minority = counts[1]  # 1 = Anomalous

    sampler = (
        RandomOverSampler(random_state=42)
        if minority < 2
        else SMOTE(random_state=42)
    )

    X_res, y_res = sampler.fit_resample(X, y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_res, y_res)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model balanced and saved to '{model_path}'.")


# ───────────────────────── Tek Etiketli Tahmin ─────────────────────────
def predict_packet(size, model_path="model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    pred = model.predict([[size]])[0]
    return "Anomalous" if pred == 1 else "Normal"


# ───────────────────── Tahmin + Olasılık (0-1) ─────────────────────
def predict_packet_with_confidence(size, model_path="model.pkl"):
    """
    Returns:
        label (str): 'Normal' or 'Anomalous'
        confidence (float): probability in [0, 1]
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    proba = model.predict_proba([[size]])[0]        # [normal_prob, anomalous_prob]
    pred_index = model.predict([[size]])[0]         # 0 or 1
    label = "Anomalous" if pred_index == 1 else "Normal"
    confidence = proba[pred_index]                  # value between 0 and 1

    return label, confidence
