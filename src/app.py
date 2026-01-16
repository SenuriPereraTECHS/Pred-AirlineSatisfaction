import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from fpdf import FPDF
import base64
import os
from dotenv import load_dotenv

# 0. Load Environment Variables
load_dotenv()
APP_TITLE = os.getenv("APP_TITLE", "Pred_Air Satisfaction Analyzer")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "best_satisfaction_model.pkl")
COLUMNS_FILENAME = os.getenv("COLUMNS_FILENAME", "model_columns.pkl")
BG_IMAGE_URL = os.getenv("BACKGROUND_IMAGE_URL", "https://images.unsplash.com/photo-1503220317375-aaad61436b1b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80")

# 1. Page Configuration
st.set_page_config(page_title=APP_TITLE, page_icon="✈️", layout="wide")

# --- UI THEME & BACKGROUND IMAGE ---
st.markdown(f"""
    <style>
    /* Fixed Background Image with optimized overlay */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(rgba(15, 23, 42, 0.75), rgba(15, 23, 42, 0.75)), 
                    url("{BG_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Global Text Visibility with stronger shadow */
    .stApp, p, label, .stSelectbox label, .stNumberInput label, .stMarkdown, h1, h2, h3 {{ 
        color: #ffffff !important; 
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,1);
    }}
    
    /* Input Box Backgrounds */
    div[data-baseweb="select"], div[data-baseweb="input"] {{
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
    }}

    /* Large Analyze Button */
    div.stButton > button {{
        background-color: #0284c7 !important;
        color: white !important;
        border-radius: 12px;
        height: 4.5em !important; 
        width: 100% !important;
        font-size: 1.6rem !important; 
        font-weight: 900 !important;
        border: 2px solid #38bdf8 !important;
        margin-top: 25px;
        text-transform: uppercase;
    }}

    /* Confidence Metric Styling */
    [data-testid="stMetricValue"] {{
        font-size: 55px !important;
        color: #38bdf8 !important;
        background-color: rgba(15, 23, 42, 0.9);
        padding: 10px 20px;
        border-radius: 15px;
        border: 2px solid #38bdf8;
    }}
    </style>
    """, unsafe_allow_html=True)

# 2. Load Assets using .env variables
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_FILENAME)
    columns = joblib.load(COLUMNS_FILENAME)
    return model, columns

model, model_columns = load_assets()

# 3. PDF Generation
def create_pdf(data, result, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"{APP_TITLE} Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Final Status: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Model Confidence: {confidence}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Submission Details:", ln=True)
    pdf.set_font("Arial", size=10)
    for key, value in data.items():
        pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# 4. Mappings
rating_map = {"Excellent (5)": 5, "Good (4)": 4, "Average (3)": 3, "Fair (2)": 2, "Poor (1)": 1, "N/A (0)": 0}
options = list(rating_map.keys())
gender_map = {"Male": 1, "Female": 0}
customer_map = {"Loyal Customer": 1, "disloyal Customer": 0}
travel_map = {"Business travel": 1, "Personal Travel": 0}
class_map = {"Business": 2, "Eco Plus": 1, "Eco": 0}

# 5. UI Layout
st.title(f"✈️ {APP_TITLE}")
st.markdown("---")

# --- STEP 1: PERSONAL DETAILS ---
st.subheader("👤 Step 1: Personal & Flight Details")
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", list(gender_map.keys()))
    age = st.number_input("Passenger Age", min_value=1, value=25)
    customer_type = st.selectbox("Customer Type", list(customer_map.keys()))
with col2:
    travel_type = st.selectbox("Type of Travel", list(travel_map.keys()))
    p_class = st.selectbox("Class", list(class_map.keys()))
    flight_dist = st.number_input("Flight Distance (km)", min_value=0, value=500)
with col3:
    dep_delay = st.number_input("Departure Delay (min)", min_value=0, value=0)
    arr_delay = st.number_input("Arrival Delay (min)", min_value=0, value=0)

# --- STEP 2: SERVICE SATISFACTION ---
st.subheader("⭐ Step 2: Service Experience Ratings")
c1, c2, c3 = st.columns(3)
with c1:
    wifi = st.selectbox("In-flight Wi-Fi Service", options, index=0)
    online_boarding = st.selectbox("Online boarding", options, index=0)
    ease_booking = st.selectbox("Ease of Online booking", options, index=0)
    online_support = st.selectbox("Online Support", options, index=0)
    seat_comfort = st.selectbox("Seat Comfort", options, index=0)
with c2:
    entertainment = st.selectbox("In-flight Entertainment", options, index=0)
    onboard_service = st.selectbox("On-board Service", options, index=0)
    leg_room = st.selectbox("Leg Room Space", options, index=0)
    cleanliness = st.selectbox("Cleanliness", options, index=0)
    food_drink = st.selectbox("Food and Drink", options, index=0)
with c3:
    checkin_service = st.selectbox("Check-in Service", options, index=0)
    handling = st.selectbox("Baggage Handling", options, index=0)
    time_convenient = st.selectbox("Departure/Arrival time convenient", options, index=0)
    gate_loc = st.selectbox("Gate location", options, index=2)

# --- ANALYSIS LOGIC ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Run AI Satisfaction Analysis", width='stretch'):
    
    input_data = {
        'Gender': gender_map[gender],
        'Customer Type': customer_map[customer_type],
        'Age': age,
        'Type of Travel': travel_map[travel_type],
        'Class': class_map[p_class],
        'Flight Distance': flight_dist,
        'Inflight wifi service': rating_map[wifi],
        'Departure/Arrival time convenient': rating_map[time_convenient],
        'Ease of Online booking': rating_map[ease_booking],
        'Gate location': rating_map[gate_loc],
        'Food and drink': rating_map[food_drink],
        'Online boarding': rating_map[online_boarding],
        'Seat comfort': rating_map[seat_comfort],
        'Inflight entertainment': rating_map[entertainment],
        'On-board service': rating_map[onboard_service],
        'Leg room service': rating_map[leg_room],
        'Baggage handling': rating_map[handling],
        'Checkin service': rating_map[checkin_service],
        'Cleanliness': rating_map[cleanliness],
        'Online support': rating_map[online_support],
        'Departure Delay in Minutes': dep_delay,
        'Arrival Delay in Minutes': arr_delay,
        'Total Delay': dep_delay + arr_delay
    }
    
    input_df = pd.DataFrame([input_data])
    
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    st.markdown("---")
    res1, res2 = st.columns([1, 2])
    
    with res1:
        status = "SATISFIED" if prediction == 1 or str(prediction).lower() == 'satisfied' else "DISSATISFIED"
        conf_val = probs[1] if status == "SATISFIED" else probs[0]
        conf_str = f"{round(conf_val*100, 1)}%"
        
        if status == "SATISFIED":
            st.success(f"### RESULT: {status}")
            st.balloons()
        else:
            st.error(f"### RESULT: {status}")
        
        st.metric(label="AI Confidence Level", value=conf_str)
        
        pdf_data = create_pdf(input_data, status, conf_str)
        st.download_button("📥 Download Analysis PDF", data=pdf_data, 
                           file_name="Airline_Satisfaction_Analysis.pdf", mime="application/pdf")

    with res2:
        fig = px.bar(x=[probs[0], probs[1]], y=['Dissatisfied', 'Satisfied'], orientation='h', 
                     color=['Dissatisfied', 'Satisfied'], color_discrete_map={'Satisfied':'#38bdf8', 'Dissatisfied':'#ef4444'},
                     labels={'x': 'Probability', 'y': 'Outcome'}, height=350)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, width='stretch')