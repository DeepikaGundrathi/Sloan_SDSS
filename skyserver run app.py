import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Skyserver_SQL2_27_2018 6_51_39 PM.csv")  
    return data

# Preprocess data
def preprocess_data(data):
    data = data.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1)
    le = LabelEncoder()
    data['class'] = le.fit_transform(data['class'])
    
    pca = PCA(n_components=3)
    ugriz = pca.fit_transform(data[['u', 'g', 'r', 'i', 'z']])
    data = pd.concat((data, pd.DataFrame(ugriz)), axis=1)
    data.rename({0: 'PCA_1', 1: 'PCA_2', 2: 'PCA_3'}, axis=1, inplace=True)
    data.drop(['u', 'g', 'r', 'i', 'z'], axis=1, inplace=True)
    
    return data

# Train and evaluate model
def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = (preds == y_test).sum().astype(float) / len(preds) * 100
    return accuracy, preds

# Main function
def main():
    st.title("Skyserver ML Model Deployment")
    
    data = load_data()
    st.write("Data Preview:")
    st.write(data.head())

    data = preprocess_data(data)
    
    scaler = MinMaxScaler()
    sdss = pd.DataFrame(scaler.fit_transform(data.drop(['mjd', 'class'], axis=1)), columns=data.drop(['mjd', 'class'], axis=1).columns)
    sdss['class'] = data['class']
    
    X = sdss.drop('class', axis=1)
    y = sdss['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    models = {
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(n_estimators=100),
        "Random Forest": RandomForestClassifier(n_estimators=10),
        "SVC": SVC()
    }
    
    st.sidebar.header("Choose Model")
    model_name = st.sidebar.selectbox("Model", list(models.keys()))
    model = models[model_name]
    
    st.write(f"Training and evaluating {model_name}...")
    accuracy, preds = train_evaluate_model(model, X_train, y_train, X_test, y_test)
    
    st.write(f"Accuracy: {accuracy:.2f}%")
    
    cm = confusion_matrix(y_test, preds)
    st.write("Confusion Matrix:")
    st.write(cm)
    
    precision = precision_score(y_test, preds, average='micro')
    recall = recall_score(y_test, preds, average='micro')
    f1 = f1_score(y_test, preds, average='micro')
    
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
    
    st.write("### Comparing all models")
    results = {}
    for name, model in models.items():
        accuracy, _ = train_evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = accuracy
    
    best_model_name = max(results, key=results.get)
    st.write(f"Best Model: {best_model_name} with accuracy: {results[best_model_name]:.2f}%")
    st.bar_chart(pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']))

if __name__ == '__main__':
    main()
