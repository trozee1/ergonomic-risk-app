import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Automotive Ergonomic Risk Classifier", layout="wide")

st.title("Automotive Workstation Ergonomic Risk Classification System")
st.markdown("### AI-Driven Occupational Hazard Assessment (Decision Tree ML Backend)")

# ==========================================
# 1. DATA COLLECTION & PREPROCESSING PIPELINE
# ==========================================
@st.cache_resource
def build_and_train_model():
    np.random.seed(42)
    n_samples = 1000
    
    # Feature dataset matching physical Ergonomic standards
    weight = np.random.uniform(2.0, 30.0, n_samples)          # kg
    reach = np.random.uniform(10.0, 90.0, n_samples)          # cm
    angle = np.random.uniform(0.0, 90.0, n_samples)           # degrees
    frequency = np.random.uniform(1.0, 20.0, n_samples)       # lifts per min
    
    # Compute ground truth risk label based on ergonomic mechanics threshold
    composite_score = (weight * 0.35) + (reach * 0.25) + (angle * 0.25) + (frequency * 0.15)
    
    # 0: Low Risk, 1: Moderate Risk, 2: High Risk
    risk_labels = []
    for score in composite_score:
        if score < 16:
            risk_labels.append(0)
        elif score < 26:
            risk_labels.append(1)
        else:
            risk_labels.append(2)
            
    df = pd.DataFrame({
        'Weight_kg': weight,
        'Reach_cm': reach,
        'Angle_deg': angle,
        'Frequency_min': frequency,
        'Risk_Level': risk_labels
    })
    
    X = df[['Weight_kg', 'Reach_cm', 'Angle_deg', 'Frequency_min']]
    y = df['Risk_Level']
    
    # 80-20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Decision Tree Classifier Initialization & Training
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions & Evaluation Metrics
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    return clf, acc, prec, rec, f1, cm

model, acc, prec, rec, f1, cm = build_and_train_model()

# ==========================================
# 2. SIDEBAR - MODEL PERFORMANCE METRICS
# ==========================================
st.sidebar.header("📊 ML Model Metrics (Section 4 Compliance)")
st.sidebar.write("**Algorithm:** Decision Tree Classifier")
st.sidebar.write("**Train-Test Split:** 80-20")
st.sidebar.metric(label="Accuracy", value=f"{acc*100:.2f}%")
st.sidebar.metric(label="Precision", value=f"{prec*100:.2f}%")
st.sidebar.metric(label="Recall", value=f"{rec*100:.2f}%")
st.sidebar.metric(label="F1-Score", value=f"{f1*100:.2f}%")

with st.sidebar.expander("Show Confusion Matrix"):
    st.write(cm)

# ==========================================
# 3. INTERACTIVE INPUT DASHBOARD
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🛠️ Workstation Input Parameters")
    zone = st.selectbox("Assembly Line Zone", ["Engine Bay Assembly", "Interior Trim Station", "Under-chassis Wiring", "Door Panel Integration"])
    weight = st.slider("Object Weight (kg)", 2.0, 30.0, 12.0, step=0.5)
    reach = st.slider("Reach Distance (cm)", 10.0, 90.0, 40.0, step=1.0)
    angle = st.slider("Torso Twisting Angle (Degrees)", 0.0, 90.0, 25.0, step=1.0)
    frequency = st.slider("Lifting Frequency (lifts / min)", 1.0, 20.0, 5.0, step=1.0)

# ==========================================
# 4. REAL-TIME AI PREDICTION & RECOMMENDATIONS
# ==========================================
with col2:
    st.subheader("⚡ Real-Time Risk Prediction")
    
    # Input vector passed into trained Decision Tree model
    input_data = np.array([[weight, reach, angle, frequency]])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    if prediction == 0:
        st.success("### Risk Classification: LOW RISK")
        st.progress(0.25)
        st.info("**Engineering Recommendation:** Workstation parameters are within safe human biomechanical thresholds. Routine monitoring advised.")
    elif prediction == 1:
        st.warning("### Risk Classification: MODERATE RISK")
        st.progress(0.65)
        st.info("**Engineering Recommendation:** Moderate musculoskeletal strain detected. Adjust table heights, optimize component layout, or limit continuous task duration.")
    else:
        st.error("### Risk Classification: HIGH RISK HAZARD")
        st.progress(1.0)
        st.info("**Engineering Recommendation:** IMMEDIATE ACTION REQUIRED! Implement mechanical assistance (e.g., spring balancers or robotic lift-assist arms) and enforce mandatory operator rotation.")

    st.markdown("---")
    st.write("**Model Confidence Breakdown:**")
    st.write(f"- Low Risk: {probabilities[0]*100:.1f}%")
    st.write(f"- Moderate Risk: {probabilities[1]*100:.1f}%")
    st.write(f"- High Risk: {probabilities[2]*100:.1f}%")
