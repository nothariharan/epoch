import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import time
import cv2  
import numpy as np  
from deepface import DeepFace  
import pandas as pd  

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Ataraxia",
    page_icon="🧠",
    layout="wide"
)

# --- NLTK Stopwords (Run once) ---
@st.cache_data
def download_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
download_stopwords()

# Gradient Themes (not working rn)
themes = {
    "Serene Mint": {
        "bg": "linear-gradient(to right, #D4E2D4, #F2F7F2)",
        "text": "#333333",
        "primary": "#4A6D4A",
        "secondary_bg": "rgba(255, 255, 255, 0.7)",
    },
    "Calm Blue": {
        "bg": "linear-gradient(to right, #E0EAFC, #CFDEF3)",
        "text": "#2C3E50",
        "primary": "#3498DB",
        "secondary_bg": "rgba(255, 255, 255, 0.7)",
    },
    "Sunset": {
        "bg": "linear-gradient(to right, #FFC3A0, #FFAFBD)",
        "text": "#50394C",
        "primary": "#FF6B6B",
        "secondary_bg": "rgba(255, 255, 255, 0.6)",
    },
    "Lavender": {
        "bg": "linear-gradient(to right, #E6E6FA, #D8BFD8)",
        "text": "#4B0082",
        "primary": "#8A2BE2",
        "secondary_bg": "rgba(255, 255, 255, 0.7)",
    },
    "Default (Dark)": {
        "bg": "#0E117", 
        "text": "#FFFFFF",
        "primary": "#00A9FF", 
        "secondary_bg": "rgba(40, 40, 40, 0.8)", 
    }
}

# --- Sidebar for Theme Selection ---
with st.sidebar:
    st.header("Settings")
    theme_name = st.selectbox("Choose a calming theme:", themes.keys(), index=0) # Default to Serene Mint

selected_theme = themes[theme_name]
BG_CSS = selected_theme["bg"]
TEXT_CSS = selected_theme["text"]
PRIMARY_CSS = selected_theme["primary"]
SECONDARY_BG_CSS = selected_theme["secondary_bg"] 

#for css injection
st.markdown(f"""
<style>
    /* Main app background */
    .appview-container {{
        background: {BG_CSS};
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    /* --- DYNAMIC TEXT --- */
    /* Headers (h1, h2, h3) */
    .appview-container h1,
    .appview-container h2,
    .appview-container h3 {{
        color: {PRIMARY_CSS} !important;
        font-weight: 600;
    }}
    
    /* Main body text */
    .appview-container p,
    .appview-container .stMarkdown,
    .appview-container .stSelectbox,
    .appview-container [data-testid="stText"] {{
        color: {TEXT_CSS} !important;
        font-size: 1.05rem;
    }}
    
    /* --- COMPONENT STYLING --- */
    
    /* Text area */
    [data-testid="stTextArea"] textarea {{
        background-color: {SECONDARY_BG_CSS} !important;
        color: {TEXT_CSS} !important;
        border: 1px solid {PRIMARY_CSS};
        border-radius: 8px;
    }}
    
    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background-color: {SECONDARY_BG_CSS};
        border-radius: 8px;
        padding: 1rem;
    }}
    [data-testid="stFileUploader"] section {{
        border-color: {PRIMARY_CSS};
        border-style: dashed;
    }}
    [data-testid="stFileUploader"] small {{
        color: {TEXT_CSS} !important;
    }}
    
    /* Button with Animation */
    [data-testid="stButton"] button {{
        background-color: {PRIMARY_CSS} !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease-in-out; /* Animation! */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    /* Button Hover Animation */
    [data-testid="stButton"] button:hover {{
        transform: translateY(-2px); /* Lifts the button */
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15); /* Bigger shadow */
        filter: brightness(1.1); /* Brighter */
    }}
    [data-testid="stButton"] button:active {{
        transform: translateY(0px); /* Pushes button back down */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Metric "Card" Styling */
    [data-testid="stMetric"] {{
        background-color: {SECONDARY_BG_CSS};
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_CSS} !important;
        opacity: 0.8 !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {PRIMARY_CSS} !important;
        font-size: 2.5rem;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {SECONDARY_BG_CSS};
        backdrop-filter: blur(5px);
    }}
    
    /* Main columns padding */
    [data-testid="stHorizontalBlock"] {{
        gap: 2rem;
    }}
    
    /* Style for the suggestion list */
    .suggestion-card {{
        background-color: {SECONDARY_BG_CSS};
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }}

</style>
""", unsafe_allow_html=True)


# --- Helper Functions (Text Analysis) ---

@st.cache_data
def get_stopwords():
    return set(stopwords.words('english'))

def clean_text(text):
    stop_words = get_stopwords()
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    # --- FIXED THE BUG HERE (cleaned_.words -> cleaned_words) ---
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

def get_suggestions(score):
    if score > 0.80:
        risk_level = "High"
        suggestions = ["It seems like you're under a lot of stress.", "It's important to talk to someone. You can connect with people who can support you by calling or texting 988 in the US and Canada, or 111 in the UK, anytime."]
    elif score > 0.50:
        risk_level = "Moderate"
        suggestions = ["You seem to be feeling some pressure.", "Try taking a 10-minute break. A short walk or some deep breathing can help.", "Consider writing down what's on your mind. This is a great journaling habit."]
    else:
        risk_level = "Low"
        suggestions = ["It's great that you are checking in with your feelings.", "Journaling is a healthy habit. Keep it up!"]
    return risk_level, suggestions

# --- Load Models (Cached for performance) ---

@st.cache_resource
def load_text_models():
    """Loads the saved text model and vectorizer from disk."""
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"Error loading text models: {e}")
        return None, None

vectorizer, model = load_text_models()

st.title("Ataraxia 🧠")
st.write("Your personal AI assistant for analyzing journal entries and expressions. Write down your thoughts or upload a picture to get gentle, supportive insights.")
st.divider()

if model is None or vectorizer is None:
    st.error("Text model files not found. Make sure 'model.pkl' and 'vectorizer.pkl' are in the same folder as 'app.py'.")
else:
    col1, col2 = st.columns(2)

    # --- COLUMN 1: Text Analysis ---
    with col1:
        st.header("Text Analysis")
        st.write("Analyze your journal entry for signs of stress.")
        
        user_text = st.text_area(
            "Write your journal entry here:",
            height=200,
            placeholder="How are you feeling today?"
        )

        if st.button("Analyze My Entry", type="primary", key="text_analyze"):
            if user_text:
                with st.spinner("Analyzing your thoughts..."):
                    cleaned_new_text = clean_text(user_text)
                    new_text_vector = vectorizer.transform([cleaned_new_text])
                    probabilities = model.predict_proba(new_text_vector)
                    stress_likelihood_score = probabilities[0][1]
                    risk, suggestions = get_suggestions(stress_likelihood_score)
                    
                    st.subheader("Your Text Analysis")
                    col1_1, col1_2 = st.columns(2)
                    col1_1.metric(label="Stress Likelihood Score", value=f"{stress_likelihood_score * 100:.2f}%")
                    col1_2.metric(label="Calculated Risk Level", value=risk)
                    
                    with st.container():
                        st.markdown('<div class="suggestion-card">', unsafe_allow_html=True)
                        st.subheader("Supportive Suggestions")
                        for s in suggestions:
                            st.write(f"🔹 {s}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze.")

    # --- COLUMN 2: Visual Analysis ---
    with col2:
        st.header("Visual Check-in")
        st.write("Detect your dominant emotion from an uploaded image.")
        
        uploaded_image = st.file_uploader(
            "Upload a picture of a face:", 
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            # To read image file buffer with OpenCV:
            bytes_data = uploaded_image.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            st.image(cv2_img, channels="BGR", caption="Your Uploaded Image", width=300) 
            with st.spinner("Analyzing your expression..."):
                try:
                    analysis = DeepFace.analyze(
                        img_path = cv2_img, 
                        actions = ['emotion'],
                        enforce_detection = True 
                    )
                    
                    result = analysis[0]
                    emotions = result['emotion']
                    
                    stress_score_sum = (
                        emotions.get('angry', 0) + 
                        emotions.get('disgust', 0) + 
                        emotions.get('fear', 0) + 
                        emotions.get('sad', 0)
                    )
                    
                    final_stress_score = min(stress_score_sum, 100.0)
                    
                    st.subheader("Your Expression Analysis")
                    st.metric(label="Calculated Stress Score", value=f"{final_stress_score:.2f}%")
                    
                    st.write("Full Emotion Breakdown:")
                    emotions_df = pd.DataFrame(emotions.items(), columns=['Emotion', 'Confidence'])
                    
                    color_map = {
                        'angry': '#FF6B6B',
                        'disgust': '#FFD166',
                        'fear': '#6A057F',  # Dark purple
                        'happy': '#06D6A0',
                        'sad': '#118AB2',
                        'surprise': '#FFC3A0',
                        'neutral': '#D3D3D3'
                    }
                    emotions_df['color'] = emotions_df['Emotion'].map(color_map)
                    
                    st.bar_chart(
                        emotions_df.set_index('Emotion'),
                        y='Confidence',
                        color='color',
                        use_container_width=True # ensure bar chart scales correctly
                    )

                except ValueError as e:
                    st.error("No face detected in the image. Please try a clearer picture.")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

