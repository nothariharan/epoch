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
import altair as alt # For stylish charts
import os      
import glob    

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
        "bg": "url('https://github.com/nothariharan/epoch/blob/main/public/serrene_mint.jpg?raw=true')",
        "text": "#333333", # Dark text for light theme
        "primary": "#4A6D4A", 
        "secondary_bg": "rgba(255, 255, 255, 0.7)", 
    },
    "Calm Blue": {
        "bg": "url('https://github.com/nothariharan/epoch/blob/main/public/calm_blue.jpg?raw=true')",
        "text": "#ffff", # Dark text for light theme
        "primary": "#3498DB", 
        "secondary_bg": "rgba(255, 255, 255, 0.7)",
    },
    "Sunset": {
        "bg": "url('https://github.com/nothariharan/epoch/blob/main/public/sunset.jpg?raw=true')",
        "text": "#ffff", # Dark text for light theme
        "primary": "#FF6B6B", 
        "secondary_bg": "rgba(255, 255, 255, 0.6)",
    },
    "Lavender": {
        "bg": "url('https://github.com/nothariharan/epoch/blob/main/public/lavender.jpg?raw=true')",
        "text": "#ffff", # Dark text for light theme
        "primary": "#8A2BE2", 
        "secondary_bg": "rgba(255, 255, 255, 0.7)",
    },
    "Default (Dark)": {
        "bg": "url('https://github.com/nothariharan/epoch/blob/main/public/dark.jpg?raw=true')",
        "text": "#FFFFFF", # White text for dark theme
        "primary": "#00A9FF", 
        "secondary_bg": "rgba(40, 40, 40, 0.8)", 
    }
}

# --- Sidebar for Theme Selection & UPDATED LOFI MUSIC ---
with st.sidebar:
    st.header("Settings")
    theme_name = st.selectbox("Choose a calming theme:", themes.keys(), index=4) # Default to Dark
    
    st.divider()
    
    st.subheader("🎵 Lofi Music")
    
    # --- NEW: Track Selection Logic ---
    LOFI_FOLDER = "lofi"
    
    # Check if the lofi folder exists
    if os.path.isdir(LOFI_FOLDER):
        mp3_files = glob.glob(os.path.join(LOFI_FOLDER, "*.mp3"))
        
        if mp3_files:
            # Initialize session state for song index
            if 'current_song_index' not in st.session_state or st.session_state.current_song_index >= len(mp3_files):
                # Pick a random song to start
                st.session_state.current_song_index = random.randint(0, len(mp3_files) - 1)
            
            current_index = st.session_state.current_song_index
            selected_song = mp3_files[current_index]
            
            # Get just the filename for display
            song_name = os.path.basename(selected_song).replace('.mp3', '').replace('_', ' ').title()
            
            # Display current track
            st.caption(f"Now playing: **{song_name}**")

            # Create columns for next/prev buttons
            col1_side, col2_side = st.columns([1, 1])
            with col1_side:
                if st.button("⏮️ Prev", use_container_width=True):
                    # Go to the previous song, wrap around if at the start
                    st.session_state.current_song_index = (current_index - 1) % len(mp3_files)
                    st.rerun() # Rerun to load the new song
            
            with col2_side:
                if st.button("Next ⏭️", use_container_width=True):
                    # Go to the next song, wrap around if at the end
                    st.session_state.current_song_index = (current_index + 1) % len(mp3_files)
                    st.rerun() # Rerun to load the new song

            # Open and play the selected local file
            # The st.audio component has its own play/pause and volume controls
            try:
                audio_file = open(selected_song, 'rb')
                audio_bytes = audio_file.read()
                st.audio(
                    audio_bytes, 
                    format="audio/mp3", 
                    loop=True
                )
            except Exception as e:
                st.error(f"Error playing '{song_name}': {e}")
        else:
            st.error(f"No .mp3 files found in the '{LOFI_FOLDER}' folder.")
            st.caption("Please add some music!")
    else:
        st.error(f"Folder not found. Please create a folder named '{LOFI_FOLDER}' and add .mp3 files.")
    # --- END OF NEW MUSIC SECTION ---


selected_theme = themes[theme_name]
BG_CSS = selected_theme["bg"]
TEXT_CSS = selected_theme["text"]
PRIMARY_CSS = selected_theme["primary"]
SECONDARY_BG_CSS = selected_theme["secondary_bg"] 

# --- NEW: Dynamic Sidebar Header Color ---
if theme_name == "Default (Dark)":
    SIDEBAR_HEADER_CSS = PRIMARY_CSS
else:
    SIDEBAR_HEADER_CSS = TEXT_CSS

# --- Custom CSS Injection (UPDATED) ---
st.markdown(f"""
<style>
    /* Main app background */
    .appview-container {{
        background: {BG_CSS};
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    /* --- DYNAMIC TEXT --- */
    /* Main headers */
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
    
    /* Main buttons */
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
    
    /* --- SIDEBAR STYLING --- */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {SECONDARY_BG_CSS};
        backdrop-filter: blur(5px);
    }}
    
    /* NEW: Style for Sidebar headers (h3) */
    [data-testid="stSidebar"] h3 {{
        color: {SIDEBAR_HEADER_CSS} !important; /* Uses new dynamic variable */
        font-size: 1.25rem !important;
    }}
    
    /* NEW: Style for Sidebar text, captions, and selectbox */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stSelectbox {{
        color: {TEXT_CSS} !important;
    }}
    
    /* NEW: Style for Audio Player */
    [data-testid="stAudio"] {{
        background-color: {SECONDARY_BG_CSS};
        border-radius: 10px;
        padding: 0.5rem 1rem 1rem 1rem; 
        margin-top: 10px; 
        border: 1px solid {PRIMARY_CSS};
    }}
    [data-testid="stAudio"] audio {{
        width: 100%;
    }}

    /* NEW: Style for Sidebar buttons (Prev/Next) */
    [data-testid="stSidebar"] [data-testid="stButton"] button {{
        background-color: {SECONDARY_BG_CSS} !important;
        color: {TEXT_CSS} !important;
        border: 1px solid {PRIMARY_CSS};
        font-weight: 600;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
        filter: brightness(0.9);
        border-color: {PRIMARY_CSS};
        transform: translateY(0px); /* Disable lift effect in sidebar */
        box-shadow: none;
    }}
    
    [data-testid="stHorizontalBlock"] {{
        gap: 2rem;
    }}
    
    /* --- NEW: Upgraded Suggestion Card Style --- */
    .suggestion-card {{
        background-color: {SECONDARY_BG_CSS};
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid {PRIMARY_CSS}; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }}
    .suggestion-header {{
        font-size: 1.25rem !important;
        font-weight: 600;
        color: {PRIMARY_CSS} !important;
        margin-bottom: 1rem;
        margin-top: -0.5rem; 
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
                    # --- All AI logic happens here, without blocking 'sleep' calls ---
                    cleaned_new_text = clean_text(user_text)
                    new_text_vector = vectorizer.transform([cleaned_new_text])
                    probabilities = model.predict_proba(new_text_vector)
                    stress_likelihood_score = probabilities[0][1]
                    
                    risk_level, suggestions = get_risk_level_and_suggestions(stress_likelihood_score)
                    risk_class = get_risk_class(risk_level)
                    
                st.subheader("Your Text Analysis")
                
                col1_1, col1_2 = st.columns(2)
                
                metric1_placeholder = col1_1.empty()
                metric2_placeholder = col1_2.empty()
                
                # --- THIS IS THE FIX ---
                # Cast to standard Python float, then to int
                final_score = float(stress_likelihood_score * 100) # Ensure it's a Python float
                for i in range(int(final_score) + 1):
                    metric1_placeholder.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Stress Likelihood Score</p>
                        <p class="metric-value {risk_class}">{i:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    # --- THIS IS THE FIX ---
                    # Also cast the denominator value to a float
                    time.sleep(0.01 / max(1.0, (float(final_score) / 20.0))) # Make animation faster for high scores

                metric1_placeholder.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Stress Likelihood Score</p>
                    <p class="metric-value {risk_class}">{final_score:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                metric2_placeholder.markdown(f"""
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
                    
                    # --- FIX: Handle if DeepFace returns a dict OR a list ---
                    if isinstance(analysis, list):
                        result = analysis[0] # Get the first face detected
                    else:
                        result = analysis # It's already the result dict
                        
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
                    
                    metric3_placeholder = col2_1.empty()
                    metric4_placeholder = col2_2.empty()

                    # --- THIS IS THE FIX ---
                    # Cast to standard Python float, then to int
                    final_stress_score = float(final_stress_score) # Ensure it's a Python float
                    for i in range(int(final_stress_score) + 1):
                        metric3_placeholder.markdown(f"""
                        <div class="metric-card">
                            <p class="metric-label">Calculated Stress Score</p>
                            <p class="metric-value {image_risk_class}">{i:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        # --- THIS IS THE FIX ---
                        # Also cast the denominator value to a float
                        time.sleep(0.01 / max(1.0, (float(final_stress_score) / 20.0))) # Make animation faster
                    
                    metric3_placeholder.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Calculated Stress Score</p>
                        <p class="metric-value {image_risk_class}">{final_stress_score:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    metric4_placeholder.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Calculated Risk Level</p>
                        <p class="metric-value {image_risk_class}" style="font-size: 2.5rem; padding-top: 10px;">{image_risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("Full Emotion Breakdown:")
                    
                    # --- STYLISH BAR CHART SECTION ---
                    emotions_df = pd.DataFrame(emotions.items(), columns=['Emotion', 'Confidence'])
                    color_map = {
                        'angry': '#FF6B6B', 'disgust': '#FFD166', 'fear': '#6A057F',
                        'happy': '#06D6A0', 'sad': '#118AB2', 'surprise': '#FFC3A0',
                        'neutral': '#D3D3D3'
                    }
                    
                    base = alt.Chart(emotions_df).properties(height=alt.Step(35))

                    bars = base.mark_bar(cornerRadius=8).encode(
                        x=alt.X('Confidence:Q', axis=None), 
                        y=alt.Y('Emotion:N', sort=None, axis=alt.Axis(
                            title=None, labels=True, domain=False, ticks=False, 
                            labelColor=TEXT_CSS, labelFontSize=12, labelPadding=5
                        )),
                        color=alt.Color('Emotion:N', legend=None, scale=alt.Scale(
                            domain=list(color_map.keys()), range=list(color_map.values())
                        )),
                        tooltip=[
                            alt.Tooltip('Emotion', title='Emotion'),
                            alt.Tooltip('Confidence', title='Confidence', format='.2f')
                        ]
                    )

                    text = bars.mark_text(align='left', baseline='middle', dx=5, color=TEXT_CSS).encode(
                        text=alt.Text('Confidence:Q', format='.2f'),
                        color=alt.value(TEXT_CSS) 
                    )
                    
                    final_chart = (bars + text).properties(background='transparent').configure_view(strokeWidth=0)
                    st.altair_chart(final_chart, use_container_width=True)
                    # --- END OF CHART SECTION ---

                except ValueError as e:
                    if "face could not be detected" in str(e).lower():
                         st.error("No face detected in the image. Please try a clearer picture.")
                    else:
                         st.error(f"An error occurred during analysis: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

