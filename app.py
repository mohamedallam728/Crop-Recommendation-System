import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
from PIL import Image

# 1. إعدادات الصفحة
st.set_page_config(page_title="Smart Crop Recommender", page_icon="🌱", layout="wide")

# 2. إعداد الشريط الجانبي (Sidebar) بأسماء التيم
st.sidebar.markdown("<h2 style='text-align: center;'>👨‍💻 Project Team</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔹 Huthyfa Moamen Marei")
st.sidebar.markdown("### 🔹 Mohamed Alaaeldin Ragab Allam")
st.sidebar.markdown("### 🔹 Serag elden Mohamed Samir Ahmed")
st.sidebar.markdown("---")
st.sidebar.success("🎓 **Supervised by:** NTI & ITIDA Summer Training")

# 3. دوال عرض اللوجوهات بخلفية زجاجية أنيقة
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def display_sleek_logo(image_path):
    base64_img = get_base64_image(image_path)
    if base64_img:
        st.markdown(
            f"""
            <div style="
                background-color: rgba(255, 255, 255, 0.85);
                padding: 15px;
                border-radius: 16px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 110px;
                margin-bottom: 20px;
                transition: transform 0.3s ease;
            ">
                <img src="data:image/png;base64,{base64_img}" style="max-height: 80px; max-width: 100%; object-fit: contain;">
            </div>
            """,
            unsafe_allow_html=True
        )

# 4. تحميل الموديل
@st.cache_resource
def load_models():
    model = joblib.load('crop_model.pkl') 
    encoder = joblib.load('label_encoder.pkl')
    return model, encoder

model, encoder = load_models()

# 5. عرض اللوجوهات والعنوان
col_logo1, col_logo2, col_spacer = st.columns([1, 1.5, 6])
with col_logo1: display_sleek_logo("logo.png")
with col_logo2: display_sleek_logo("1694171296itida-1024x502.png")

st.title("🌾 Smart Crop Recommendation System")
st.markdown("### 🤖 Applied Machine Learning & AI Project")
st.markdown("---")

# 6. تقسيم الموقع لـ 4 أقسام
tab1, tab2, tab3, tab4 = st.tabs([
    "🌱 Prediction & AI Logic", 
    "📊 Models Comparison", 
    "🏆 Final Decision",
    "📈 Data Insights"
])

# ================= القسم الأول: التوقع وشرح الذكاء الاصطناعي =================
with tab1:
    st.header("Adjust Soil and Environmental Parameters")
    st.info("💡 Use the sliders to adjust the parameters. The AI will recommend the best crop instantly.")
    
    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("Nitrogen (N)", 0.0, 140.0, 90.0)
        P = st.slider("Phosphorus (P)", 0.0, 145.0, 42.0)
        K = st.slider("Potassium (K)", 0.0, 205.0, 43.0)
        temperature = st.slider("Temperature (°C)", 5.0, 45.0, 20.8)
    with col2:
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 82.0)
        ph = st.slider("pH Level", 3.5, 10.0, 6.5)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 202.9)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Predict Suitable Crop", use_container_width=True):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        crop_name = encoder.inverse_transform(prediction)[0]
        st.success(f"## ✅ The most suitable crop for cultivation is: **{crop_name.upper()}**")
        
        # Explainable AI: Feature Importance
        st.markdown("---")
        st.subheader("🧠 How did the AI make this decision?")
        try:
            importance = model.feature_importances_
            features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
            importance_df = pd.DataFrame({'Feature': features, 'Impact (%)': importance * 100})
            importance_df = importance_df.sort_values(by='Impact (%)', ascending=False).set_index('Feature')
            
            st.markdown("This chart shows which environmental factors influenced the model's decision the most:")
            st.bar_chart(importance_df)
        except:
            st.warning("Feature importance is not available for this specific model type.")

# ================= القسم الثاني: مقارنة الموديلات =================
with tab2:
    st.header("📊 Algorithm Accuracy Comparison")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label="Decision Tree", value="90.5%")
    c2.metric(label="SVM Model", value="97.8%")
    c3.metric(label="Random Forest", value="99.0%")
    c4.metric(label="XGBoost", value="99.3%", delta="Highest Accuracy")
    
    st.markdown("---")
    st.subheader("🖼️ Confusion Matrices")
    
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**1. Decision Tree**")
        try: st.image(Image.open("tree_cm.png"), use_container_width=True)
        except: pass
        
        st.markdown("**3. Random Forest**")
        try: st.image(Image.open("rf_cm.png"), use_container_width=True)
        except: pass

    with img_col2:
        st.markdown("**2. SVM Model**")
        try: st.image(Image.open("svm_cm.png"), use_container_width=True)
        except: pass
        
        st.markdown("**4. XGBoost Model**")
        try: st.image(Image.open("xgb_cm.png"), use_container_width=True)
        except: pass

# ================= القسم الثالث: القرار النهائي =================
with tab3:
    st.header("🏆 Winning Model")
    st.success("## Based on the comprehensive analysis, the winning model is: **XGBoost** 🎉")
    
    st.markdown(
        """
        ### 🔍 Justification for Selection:
        * 🎯 **Maximum Precision:** Consistently achieved the highest prediction accuracy (>99%).
        * 📉 **Robust Error Handling:** Demonstrated superior capability in distinguishing between crops with highly overlapping required parameters.
        * 🚀 **Algorithmic Efficiency:** Utilized Gradient Boosting effectively to capture non-linear complex relationships without overfitting.
        """
    )

# ================= القسم الرابع: تحليل البيانات =================
with tab4:
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.info("A quick look at the dataset structure and relationships before training the models.")
    
    try:
        df = pd.read_csv('Crop_recommendation.csv')
        st.markdown("**Sample of the Raw Dataset:**")
        st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("---")
        st.markdown("**🌡️ Scatter Plot: Temperature vs Rainfall across different crops**")
        # رسم العلاقة بين الحرارة والمطر بناءً على المحصول
        st.scatter_chart(df, x='temperature', y='rainfall', color='label', height=500)
    except:
        st.warning("⚠️ Please place the 'Crop_recommendation.csv' file in the project folder to view the interactive data charts.")