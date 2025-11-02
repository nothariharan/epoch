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
import random  # Added for dynamic spinner messages
import altair as alt # <-- NEW IMPORT FOR STYLISH CHARTS

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="MindfulAI",
    page_icon="🧠",
    layout="wide"
)

# --- NLTK Stopwords Download ---
@st.cache_data
def download_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
download_stopwords()

# --- THEMES (No change) ---
themes = {
    "Serene Mint": {
        "bg": "linear-gradient(to right, #D4E2D4, #F2F7F2)",
        "text": "#333333",
        "primary": "#4A6D4A", # Green (for Low Risk)
        "secondary_bg": "rgba(255, 255, 255, 0.7)", 
    },
    "Calm Blue": {
        "bg": "linear-gradient(to right, #E0EAFC, #CFDEF3)",
        "text": "#2C3E50",
        "primary": "#3498DB", # Blue (for Low Risk)
        "secondary_bg": "rgba(255, 255, 255, 0.7)",
    },
    "Sunset": {
        "bg": "linear-gradient(to right, #FFC3A0, #FFAFBD)",
        "text": "#50394C",
        "primary": "#FF6B6B", # Red (for Low Risk)
        "secondary_bg": "rgba(255, 255, 255, 0.6)",
    },
    "Lavender": {
        "bg": "linear-gradient(to right, #E6E6FA, #D8BFD8)",
        "text": "#4B0082",
        "primary": "#8A2BE2", # Purple (for Low Risk)
        "secondary_bg": "rgba(255, 255, 255, 0.7)",
    },
    "Default (Dark)": {
        "bg": "#0E117", 
        "text": "#FFFFFF",
        "primary": "#00A9FF", # Blue (for Low Risk)
        "secondary_bg": "rgba(40, 40, 40, 0.8)", 
    }
}

# --- Sidebar for Theme Selection & NEW LOFI MUSIC ---
with st.sidebar:
    st.header("Settings")
    theme_name = st.selectbox("Choose a calming theme:", themes.keys(), index=0)
    
    st.divider()
    
    st.subheader("🎵 Lofi Music")
    st.caption("Music for a calm session. Press play to start.")
    st.audio(
        "https://raw.githubusercontent.com/serp-ai/code-interpreter-assets/main/lofi-study-112-bpm.mp3", 
        format="audio/mp3", 
        loop=True
    )

selected_theme = themes[theme_name]
BG_CSS = selected_theme["bg"]
TEXT_CSS = selected_theme["text"]
PRIMARY_CSS = selected_theme["primary"]
SECONDARY_BG_CSS = selected_theme["secondary_bg"] 

# --- Custom CSS Injection (UPDATED) ---
st.markdown(f"""
<style>
    /* Main app background */
    .appview-container {{
        background: {BG_CSS};
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    /* --- DYNAMIC TEXT --- */
    .appview-container h1,
    .appview-container h2,
    .appview-container h3 {{
        color: {PRIMARY_CSS} !important;
        font-weight: 600;
    }}
    
    .appview-container p,
    .appview-container .stMarkdown,
    .appview-container .stSelectbox,
    .appview-container [data-testid="stText"] {{
        color: {TEXT_CSS} !important;
        font-size: 1.05rem;
    }}
    
    /* --- COMPONENT STYLING --- */
    [data-testid="stTextArea"] textarea {{
        background-color: {SECONDARY_BG_CSS} !important;
        color: {TEXT_CSS} !important;
        border: 1px solid {PRIMARY_CSS};
        border-radius: 8px;
    }}
    
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
    
    /* Button with Animation (No change) */
    [data-testid="stButton"] button {{
        background-color: {PRIMARY_CSS} !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease-in-out; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    [data-testid="stButton"] button:hover {{
        transform: translateY(-2px); 
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15); 
        filter: brightness(1.1); 
    }}
    [data-testid="stButton"] button:active {{
        transform: translateY(0px); 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* --- Custom Metric Styling (No change) --- */
    .metric-card {{
        background-color: {SECONDARY_BG_CSS};
        border-radius: 10px;
        margin-bottom: 25px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        height: 100%; 
    }}
    .metric-label {{
        color: {TEXT_CSS} !important;
        opacity: 0.8 !important;
        font-size: 1rem; 
        margin-bottom: -5px; 
    }}
    .metric-value {{
        font-size: 3rem !important;
        font-weight: 600;
        margin-top: 0px;
    }}
    
    /* --- Dynamic Color Classes (No change) --- */
    .metric-value.risk-low {{
        color: {PRIMARY_CSS} !important; 
    }}
    .metric-value.risk-moderate {{
        color: #E8A900 !important; 
    }}
    .metric-value.risk-high {{
        color: #D14343 !important; 
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {SECONDARY_BG_CSS};
        backdrop-filter: blur(5px);
    }}
    /* NEW: Style for Sidebar text */
    [data-testid="stSidebar"] h3 {{
        color: {PRIMARY_CSS} !important;
        font-size: 1.25rem !important;
    }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] .stCaption {{
        color: {TEXT_CSS} !important;
    }}
    /* NEW: Style for Audio Player */
    [data-testid="stAudio"] {{
        background-color: {SECONDARY_BG_CSS};
        border-radius: 10px;
        padding: 0.5rem 1rem 1rem 1rem; /* Add some padding */
        margin-top: -10px; /* Pull it up closer to the caption */
        border: 1px solid {PRIMARY_CSS};
    }}
    [data-testid="stAudio"] audio {{
        width: 100%;
    }}
    
    [data-testid="stHorizontalBlock"] {{
        gap: 2rem;
    }}
    
    /* --- NEW: Upgraded Suggestion Card Style --- */
    .suggestion-card {{
        background-color: {SECONDARY_BG_CSS};
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid {PRIMARY_CSS}; /* Accent border */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }}
    .suggestion-header {{
        font-size: 1.25rem !important;
        font-weight: 600;
        color: {PRIMARY_CSS} !important;
        margin-bottom: 1rem;
        margin-top: -0.5rem; /* Pull header up a bit */
    }}
    .suggestion-item {{
        font-size: 1.05rem !important; 
        line-height: 1.6;         
        margin-bottom: 0.5rem;         
        color: {TEXT_CSS} !important;
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
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

def get_risk_level_and_suggestions(score):
    if score > 0.80:
        risk_level = "High"
        # --- RE-ADDED INDIA-SPECIFIC HELPLINES ---
        suggestions = [
            "It seems like you're under a lot of stress.",
            "It's important to talk to someone. You can connect with 24/7 helplines in India, like the **Kiran Helpline at 1800-599-0019** or the **Vandrevala Foundation at 9999666555**."
        ]
    elif score > 0.50:
        risk_level = "Moderate"
        suggestions = ["You seem to be feeling some pressure.", "Try taking a 10-minute break. A short walk or some deep breathing can help.", "Consider writing down what's on your mind. This is a great journaling habit."]
    else:
        risk_level = "Low"
        suggestions = ["It's great that you are checking in with your feelings.", "Journaling is a healthy habit. Keep it up!"]
    return risk_level, suggestions

def get_risk_class(risk_level):
    if risk_level == "High":
        return "risk-high"
    if risk_level == "Moderate":
        return "risk-moderate"
    return "risk-low"

# --- Load Models (Cached for performance - No change) ---
@st.cache_resource
def load_text_models():
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

# --- NEW: Dynamic Spinner Messages ---
calm_messages = [
    "Taking a deep breath...",
    "Analyzing your thoughts with care...",
    "Finding insights for you...",
    "Hold on, just a moment...",
    "Processing with a calm mind...",
]

# --- UI ---
st.title("MindfulAI 🧠")
st.write("Your personal AI assistant for analyzing journal entries and expressions. Write down your thoughts or upload a picture to get gentle, supportive insights.")
st.divider()

if model is None or vectorizer is None:
    st.error("Text model files not found. Make sure 'model.pkl' and 'vectorizer.pkl' are in the same folder as 'app.py'.")
else:
    col1, col2 = st.columns(2)

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
                with st.spinner(random.choice(calm_messages)):
                    cleaned_new_text = clean_text(user_text)
                    new_text_vector = vectorizer.transform([cleaned_new_text])
                    probabilities = model.predict_proba(new_text_vector)
                    stress_likelihood_score = probabilities[0][1]
                    
                    risk_level, suggestions = get_risk_level_and_suggestions(stress_likelihood_score)
                    risk_class = get_risk_class(risk_level)
                    
                    st.subheader("Your Text Analysis")
                    
                    col1_1, col1_2 = st.columns(2)
                    
                    with col1_1:
                        metric_placeholder_1 = st.empty()
                    with col1_2:
                        metric_placeholder_2 = st.empty()
                    
                    final_score = stress_likelihood_score * 100
                    for i in range(int(final_score) + 1):
                        metric_placeholder_1.markdown(f"""
                        <div class="metric-card">
                            <p class="metric-label">Stress Likelihood Score</p>
                            <p class="metric-value {risk_class}">{i:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.01) 
                    
                    metric_placeholder_1.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Stress Likelihood Score</p>
                        <p class="metric-value {risk_class}">{final_score:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    metric_placeholder_2.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Calculated Risk Level</p>
                        <p class="metric-value {risk_class}" style="font-size: 2.5rem; padding-top: 10px;">...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.5) 
                    metric_placeholder_2.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Calculated Risk Level</p>
                        <p class="metric-value {risk_class}" style="font-size: 2.5rem; padding-top: 10px;">{risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    suggestions_html = f'<div class="suggestion-card"><p class="suggestion-header">Supportive Suggestions</p>'
                    for s in suggestions:
                        suggestions_html += f'<div class="suggestion-item">🔹 {s}</div>'
                    suggestions_html += '</div>'
                    
                    st.markdown(suggestions_html, unsafe_allow_html=True)
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
            bytes_data = uploaded_image.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            st.image(cv2_img, channels="BGR", caption="Your Uploaded Image", width=300) 
            
            with st.spinner(random.choice(calm_messages)):
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
                    
                    if final_stress_score > 80.0:
                        image_risk_level = "High"
                    elif final_stress_score > 50.0:
                        image_risk_level = "Moderate"
                    else:
                        image_risk_level = "Low"
                    
                    image_risk_class = get_risk_class(image_risk_level)
                    
                    st.subheader("Your Expression Analysis")
                    
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        img_metric_placeholder_1 = st.empty()
                    with col2_2:
                        img_metric_placeholder_2 = st.empty()

                    for i in range(int(final_stress_score) + 1):
                        img_metric_placeholder_1.markdown(f"""
                        <div class="metric-card">
                            <p class="metric-label">Calculated Stress Score</p>
                            <p class="metric-value {image_risk_class}">{i:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.01) 
                    
                    img_metric_placeholder_1.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Calculated Stress Score</p>
                        <p class="metric-value {image_risk_class}">{final_stress_score:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    img_metric_placeholder_2.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Calculated Risk Level</p>
                        <p class="metric-value {image_risk_class}" style="font-size: 2.5rem; padding-top: 10px;">...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.5) 
                    img_metric_placeholder_2.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Calculated Risk Level</p>
                        <p class="metric-value {image_risk_class}" style="font-size: 2.5rem; padding-top: 10px;">{image_risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("Full Emotion Breakdown:")
                    
                    # --- THIS IS THE NEW, STYLISH BAR CHART SECTION ---
                    
                    # 1. Convert data to DataFrame
                    emotions_df = pd.DataFrame(emotions.items(), columns=['Emotion', 'Confidence'])
                    
                    # 2. Define the color map
                    color_map = {
                        'angry': '#FF6B6B',
                        'disgust': '#FFD166',
                        'fear': '#6A057F',
                        'happy': '#06D6A0',
                        'sad': '#118AB2',
                        'surprise': '#FFC3A0',
                        'neutral': '#D3D3D3'
                    }
                    
                    # 3. Create the base chart
                    base = alt.Chart(emotions_df).properties(
                        # --- UPDATED: Set height by "step" to add padding ---
                        height=alt.Step(35) # This gives each bar 35px of space
                    )

                    # 4. Create the bars
                    bars = base.mark_bar(
                        cornerRadius=8, 
                        # height=20 <-- REMOVED this fixed height
                    ).encode(
                        x=alt.X('Confidence:Q', axis=None), # Hide X-axis
                        y=alt.Y('Emotion:N', sort=None, axis=alt.Axis(
                            title=None, # Remove "Emotion" title
                            labels=True,
                            domain=False, # Remove axis line
                            ticks=False, # Remove tick marks
                            labelColor=TEXT_CSS, # Use theme color for labels
                            labelFontSize=12,
                            labelPadding=5
                        )),
                        # Use the color map for the bars
                        color=alt.Color('Emotion:N',
                            legend=None, # Hide the color legend
                            scale=alt.Scale(
                                domain=list(color_map.keys()),
                                range=list(color_map.values())
                            )
                        ),
                        # Add tooltips for hover
                        tooltip=[
                            alt.Tooltip('Emotion', title='Emotion'),
                            alt.Tooltip('Confidence', title='Confidence', format='.2f')
                        ]
                    )

                    # 5. Create the text labels for the bars
                    text = bars.mark_text(
                        align='left',
                        baseline='middle',
                        dx=5,  # Nudge text 5px to the right of the bar
                        color=TEXT_CSS # Use theme color for text
                    ).encode(
                        text=alt.Text('Confidence:Q', format='.2f'),
                        color=alt.value(TEXT_CSS) # Override color to be theme-based
                    )
                    
                    # 6. Combine the bars and text
                    final_chart = (bars + text).properties(
                        background='transparent' # Make background transparent
                    ).configure_view(
                        strokeWidth=0 # Remove the chart border
                    )

                    # 7. Display the chart in Streamlit
                    st.altair_chart(final_chart, use_container_width=True)
                    # --- END OF NEW CHART SECTION ---

                except ValueError as e:
                    st.error("No face detected in the image. Please try a clearer picture.")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

