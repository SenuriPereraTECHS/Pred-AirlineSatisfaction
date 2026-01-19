# Pred_Air Satisfaction Analyzer
✈️ Pred_Air Satisfaction Analyzer
📖 Description
    Pred_Air Satisfaction Analyzer is an intelligent analytical system designed to predict and analyze passenger satisfaction levels for airline operators.

    Users can input detailed flight data and service ratings to receive an immediate AI-driven sentiment analysis. The system leverages a Random Forest Classifier to evaluate 22+ variables, providing airlines with actionable insights into passenger loyalty and service gaps.

📝 Overview
    This project provides a professional-grade interface for airline analysts to gauge passenger sentiment.

    When user data is submitted, the system processes the information through a sophisticated machine learning pipeline. The final output includes a confidence-weighted prediction, a probability distribution chart, and a downloadable PDF report for official record-keeping.

✨ Features
    AI Sentiment Prediction: Utilizes an optimized Random Forest model for high-accuracy satisfaction forecasting.

    Interactive Analytics: Features real-time Probability Charts for individual passenger outcome profiling.

    Feature Importance: A dedicated sidebar insight tool showing the top global drivers of satisfaction.

    Automated Reporting: Instant PDF generation and download for every analyzed passenger.

    Professional UI: High-contrast, dark-mode interface with an integrated airplane background and fluid navigation.

🤖 System Architecture
    Pred_Air Satisfaction Analyzer employs a structured pipeline to ensure data fed to the model matches the training environment perfectly.

    1. Preprocessing Engine 🧩
    Label Encoding: Converts categorical text (e.g., "Business Class") into model-ready numerical values (e.g., 2).

    Feature Engineering: Automatically calculates the Total Delay feature by summing Departure and Arrival metrics.

    Data Alignment: Ensures the input vector precisely matches the 23-column structure expected by the model.

    2. Prediction Engine 🔍
    Classification: Determines the final status (Satisfied vs. Dissatisfied).

    Confidence Scoring: Extracts the predict_proba values to show exactly how certain the AI is about the result.

    3. Reporting Engine 💡
    Visualization: Renders Plotly-based probability distributions and success animations.

    Document Generation: Uses FPDF to compile all inputs and results into a formal PDF report.

💻 Technology Stack
    Core: Python 3.12

    Backend: Scikit-learn, Pandas, Joblib

    Frontend: Streamlit

    Visualization: Plotly Express

    Utilities: FPDF, Python-dotenv

🚀 Getting Started
✅ Prerequisites
    Python 3.10 or higher

    pip (Python package installer)

📥 Installation
    Clone the repository:

    Bash

    git clone https://github.com/your-username/Pred-AirlineSatisfaction.git
    cd Pred-AirlineSatisfaction
    
    Install dependencies:

    Bash

    pip install -r requirements.txt

▶️ How to Run
    Bash

    streamlit run app.py
⚠️ Disclaimer
    This tool is for informational and analytical purposes only. While the AI provides high-accuracy predictions based on historical data, results should be used as a supplement to—not a replacement for—direct customer feedback and professional business intuition.


🛡️ License

    Distributed under the MIT License. See `LICENSE` for more information.

    Copyright (c) 2026 [SenuriPerera]


