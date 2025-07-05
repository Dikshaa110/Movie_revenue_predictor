# ======================
# 1. PACKAGE VERIFICATION
# ======================
import os
import sys
import subprocess
import pkg_resources

# List of required packages
REQUIRED_PACKAGES = {
    'joblib==1.3.2',
    'pandas==2.1.4',
    'matplotlib==3.8.2',
    'scikit-learn==1.3.2',
    'streamlit==1.29.0'
}

def install_missing_packages():
    """Install missing packages automatically"""
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = {pkg.split('==')[0] for pkg in REQUIRED_PACKAGES} - installed
    
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing],
            stdout=subprocess.DEVNULL
        )

# Run before any other imports
install_missing_packages()

# ======================
# 2. MAIN IMPORTS (Protected)
# ======================
try:
    import streamlit as st
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_absolute_error
    
except ImportError as e:
    print(f"Critical import error: {e}")
    sys.exit(1)

# ======================
# 3. MODEL LOADING (Protected)
# ======================
try:
    model_bundle = joblib.load("stacked_movie_revenue_model.joblib")
    model = model_bundle["model"]
    expected_features = model_bundle["features"]
except Exception as e:
    st.error(f"""
    ## ⚠️ Model Loading Failed
    **Error:** {str(e)}
    
    Please ensure:
    1. `stacked_movie_revenue_model.joblib` exists in your project
    2. The file is not corrupted
    3. You have sufficient permissions
    """)
    st.stop()
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model_bundle = joblib.load("stacked_movie_revenue_model.joblib")
model = model_bundle["model"]
expected_features = model_bundle["features"]

# Configure page
st.set_page_config(
    page_title="Movie Revenue Predictor PRO",
    page_icon="",
    layout="centered"
)

# Custom CSS for blue form with proper contrast
st.markdown("""
<style>
    :root {
        --primary: #1E88E5;
        --primary-dark: #0D47A1;
        --secondary: #FFC107;
        --background: #FFFFFF;
        --card: #F5F9FF;
        --text-dark: #000000;
        --text-light: #FFFFFF;
    }
    
    .stApp {
        background-color: var(--background);
    }
    
    /* Blue form container */
    .stExpander {
        background-color: var(--primary-dark);
        border-radius: 10px;
        padding: 20px;
        color: var(--text-light) !important;
    }
    
    /* White content areas */
    .stMetric, .css-1aumxhk, .stTable {
        background-color: var(--background) !important;
        color: var(--text-dark) !important;
        border-radius: 8px;
    }
    
    /* White text on blue form */
    .stExpander label, .stExpander .stMarkdown, 
    .stExpander .stNumberInput, .stExpander .stSelectbox,
    .stExpander .stSlider {
        color: var(--text-light) !important;
    }
    
    /* Input fields styling */
    .stNumberInput>div>div>input, .stSelectbox>div>div>select,
    .stTextInput>div>div>input {
        background-color: rgba(255,255,255,0.1) !important;
        color: var(--text-light) !important;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Slider track */
    .stSlider .st-cc {
        background: rgba(255,255,255,0.3) !important;
    }
    
    /* Slider thumb */
    .stSlider .st-cd {
        background: var(--text-light) !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: var(--text-light) !important;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Results card */
    .revenue-card {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: var(--text-light);
        border-radius: 10px;
        padding: 20px;
        margin: 16px 0;
    }
    
    /* Table styling */
    .stTable {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div style="text-align: center; margin-bottom: 24px;">
    <h1 style="color: var(--primary-dark);">Movie Revenue Predictor</h1>
    <p style="color: var(--text-dark);">
        Predict your movie's box office potential with AI
    </p>
</div>
""", unsafe_allow_html=True)

# Blue Form Section
with st.expander("📋 Enter Movie Details", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.number_input(
            "💰 Budget (USD)",
            min_value=1000000,
            max_value=500000000,
            value=50000000,
            step=1000000
        )
        popularity = st.slider(
            "📈 Popularity Score",
            0.0, 150.0, 50.0
        )
        runtime = st.slider(
            "⏱ Runtime (minutes)",
            30, 240, 120
        )
        
    with col2:
        vote_average = st.slider(
            "⭐ Vote Average",
            0.0, 10.0, 6.0,
            step=0.1
        )
        vote_count = st.slider(
            "🗳 Vote Count",
            0, 50000, 1000
        )
        release_year = st.slider(
            "📆 Release Year",
            1950, 2025, 2020
        )
    
    status = st.selectbox(
        " Status",
        ["Released", "Post Production", "Rumored", "In Production"]
    )
    original_language = st.selectbox(
        "🗣 Original Language",
        ["en", "fr", "es", "hi", "ja", "de", "zh", "ru", "it", "ko", "pt", "cn"]
    )
    main_genre = st.selectbox(
        "🎞 Main Genre",
        ["Action", "Adventure", "Comedy", "Drama", "Fantasy", 
         "Horror", "Thriller", "Romance", "Animation", "Science Fiction"]
    )

# Create input dataframe
input_data = {
    'budget': [budget],
    'popularity': [popularity],
    'runtime': [runtime],
    'vote_average': [vote_average],
    'vote_count': [vote_count],
    'release_year': [release_year],
    'status': [status],
    'original_language': [original_language],
    'main_genre': [main_genre]
}
input_df = pd.DataFrame(input_data)[expected_features]

# Prediction Button
if st.button(" Predict Revenue", type="primary"):
    prediction = model.predict(input_df)[0]
    
    # Format prediction
    if prediction < 1_000_000:
        pred_text = f"${prediction:,.0f}"
    elif prediction < 1_000_000_000:
        pred_text = f"${prediction/1_000_000:,.1f} million"
    else:
        pred_text = f"${prediction/1_000_000_000:,.2f} billion"
    
    # Calculate metrics
    profit = prediction - budget
    roi = (profit / budget) * 100
    
    # Results Card (blue background with white text)
    st.markdown(f"""
    <div class="revenue-card">
        <h2 style="color: white; margin-top: 0;">Predicted Revenue</h2>
        <h1 style="color: white; margin-bottom: 8px;">{pred_text}</h1>
        <div style="display: flex; justify-content: space-between; color: white;">
            <span>Budget: ${budget/1_000_000:,.1f}M</span>
            <span>ROI: {roi:+.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Analytics Section (white background with black text)
    st.subheader(" Financial Analytics")
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Budget", f"${budget/1_000_000:,.1f}M")
    with cols[1]:
        st.metric("Estimated Profit", f"${profit/1_000_000:,.1f}M")
    with cols[2]:
        st.metric("ROI", f"{roi:+.1f}%")
    
    # Visualization (white background)
    st.subheader("📈 Revenue vs Budget")
    fig, ax = plt.subplots()
    bars = ax.bar(
        ["Budget", "Predicted Revenue"],
        [budget, prediction],
        color=["#1E88E5", "#0D47A1"]
    )
    ax.set_ylabel("Amount (USD)")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height/1_000_000:,.1f}M',
                ha='center', va='bottom')
    st.pyplot(fig)
    
    # Key Factors (white background)
    st.subheader("Key Influencing Factors")
    factors = {
        "Budget": f"${budget/1_000_000:,.1f}M",
        "Popularity": f"{popularity:.1f}",
        "Genre": main_genre,
        "Vote Average": f"{vote_average:.1f}/10",
        "Runtime": f"{runtime} mins"
    }
    st.table(pd.DataFrame.from_dict(factors, orient='index', columns=['Value']))
    
    if prediction > 1_000_000_000:
        st.balloons()
        st.success("🌟 Blockbuster potential! This prediction exceeds $1 billion")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 48px; color: #666; font-size: 14px;">
    <hr style="border: 0.5px solid #E0E0E0; margin-bottom: 16px;">
    <p>Movie Revenue Predictor • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)