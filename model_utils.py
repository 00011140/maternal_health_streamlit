import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    return pd.read_csv("data/maternal_health_risk.csv")

def preprocess_data(df):
    # Encode target
    le = LabelEncoder()
    df["RiskLevel"] = le.fit_transform(df["RiskLevel"])

    X = df.drop("RiskLevel", axis=1)
    y = df["RiskLevel"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, le, scaler

def train_models(X_train_scaled, X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000, multi_class="multinomial")
    log_reg.fit(X_train_scaled, y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    return log_reg, knn, rf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report
