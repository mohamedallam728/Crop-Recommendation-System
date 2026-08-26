import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
from PIL import Image

# 1. إعدادات الصفحة
st.set_page_config(page_title="Smart Crop & Yield Recommender", page_icon="🌱", layout="wide")

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

# 4. تحميل الموديلات (التصنيف والانحدار) مع الـ Caching
@st.cache_resource
def load_all_models():
    crop_model = joblib.load('crop_model.pkl') 
    crop_encoder = joblib.load('label_encoder.pkl')
    try:
        yield_model = joblib.load('yield_model.pkl') # موديل الانحدار الجديد
    except:
        yield_model = None
    return crop_model, crop_encoder, yield_model

model, encoder, yield_model = load_all_models()

# 5. عرض اللوجوهات والعنوان الرئيسي
col_logo1, col_logo2, col_spacer = st.columns([1, 1.5, 6])
with col_logo1: display_sleek_logo("logo.png")
with col_logo2: display_sleek_logo("1694171296itida-1024x502.png")

st.title("🌾 Smart Agricultural Assistant System")
st.markdown("### 🤖 Applied Machine Learning & AI Project (Crop Recommendation & Yield Prediction)")
st.markdown("---")

# 6. تقسيم الموقع لـ Tabs متكاملة (تشمل التصنيف والانحدار والتحليلات)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌱 Crop Recommendation", 
    "📈 Yield Prediction (Regression)", 
    "📊 Models Comparison", 
    "🏆 Final Decision",
    "📈 Data Insights"
])

# ================= التاب الأول: توصية المحصول (Classification) =================
with tab1:
    st.header("Adjust Soil and Environmental Parameters for Classification")
    st.info("💡 Use the sliders to adjust the parameters. The AI will recommend the best crop instantly.")
    
    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("Nitrogen (N)", 0.0, 140.0, 90.0, key="c_n")
        P = st.slider("Phosphorus (P)", 0.0, 145.0, 42.0, key="c_p")
        K = st.slider("Potassium (K)", 0.0, 205.0, 43.0, key="c_k")
        temperature = st.slider("Temperature (°C)", 5.0, 45.0, 20.8, key="c_temp")
    with col2:
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 82.0, key="c_hum")
        ph = st.slider("pH Level", 3.5, 10.0, 6.5, key="c_ph")
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 202.9, key="c_rain")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Predict Suitable Crop", use_container_width=True, key="btn_crop"):
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

# ================= التاب الثاني: توقع الإنتاجية (Regression) =================
with tab2:
    st.header("Predict Expected Crop Yield (hg/ha)")
    st.info("💡 Estimate the harvest amount based on soil and climate conditions using our optimized Regression model.")
    
    y_N = st.number_input("Nitrogen (N) for Yield", 0.0, 140.0, 50.0, key="y_n")
    y_P = st.number_input("Phosphorus (P) for Yield", 0.0, 145.0, 50.0, key="y_p")
    y_K = st.number_input("Potassium (K) for Yield", 0.0, 205.0, 50.0, key="y_k")
    y_temp = st.number_input("Temperature (°C) for Yield", 5.0, 45.0, 25.0, key="y_temp")
    y_hum = st.number_input("Humidity (%) for Yield", 10.0, 100.0, 70.0, key="y_hum")
    y_ph = st.number_input("pH Level for Yield", 3.5, 10.0, 6.5, key="y_ph")
    y_rain = st.number_input("Rainfall (mm) for Yield", 20.0, 300.0, 100.0, key="y_rain")
    
    if st.button("Calculate Expected Yield", use_container_width=True, key="btn_yield"):
        if yield_model is not None:
            input_data = np.array([[y_N, y_P, y_K, y_temp, y_hum, y_ph, y_rain]])
            predicted_yield = yield_model.predict(input_data)[0]
            st.metric(label="Estimated Yield Production", value=f"{predicted_yield:,.2f} hg/ha")
            st.info("💡 Note: hg/ha stands for Hectogram per hectare.")
        else:
            st.error("⚠️ Regression model file ('yield_model.pkl') is missing. Please upload it to your repository.")

# ================= التاب الثالث: مقارنة الموديلات =================
with tab3:
    st.header("📊 Algorithm Accuracy Comparison (Classification)")
    
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

# ================= التاب الرابع: القرار النهائي =================
with tab4:
    st.header("🏆 Winning Model")
    st.success("## Based on comprehensive analysis, the winning model is: **XGBoost** 🎉")
    
    st.markdown(
        """
        ### 🔍 Justification for Selection:
        * 🎯 **Maximum Precision:** Consistently achieved highest prediction accuracy (>99%).
        * 📉 **Robust Error Handling:** Demonstrated superior capability in distinguishing between crops with overlapping parameters.
        * 🚀 **Algorithmic Efficiency:** Utilized Gradient Boosting effectively to capture non-linear complex relationships without overfitting.
        """
    )

# ================= التاب الخامس: تحليل البيانات =================
with tab5:
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.info("A quick look at the dataset structure and relationships before training the models.")
    
    try:
        df = pd.read_csv('Crop_recommendation.csv')
        st.markdown("**Sample of the Raw Dataset:**")
        st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("---")
        st.markdown("**🌡️ Scatter Plot: Temperature vs Rainfall across different crops**")
        st.scatter_chart(df, x='temperature', y='rainfall', color='label', height=500)
    except:
        st.warning("⚠️ Please place the 'Crop_recommendation.csv' file in the project folder to view the interactive data charts.")

# التذييل (Footer)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 Smart Crop Recommendation System | Developed by NTI Trainees</p>", unsafe_allow_html=True)