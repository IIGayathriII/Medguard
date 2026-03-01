"""
Streamlit Web UI for Hybrid Medical Chatbot
Combines Bayesian Network (hallucination-free predictions) with Gemini API (natural language)
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import json
import re
import time
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from hybrid_chatbot import HybridMedicalChatbot
import plotly.graph_objects as go
import pandas as pd

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced user experience
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1565C0;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Symptom chips */
    .symptom-chip {
        display: inline-block;
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Prediction cards */
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #1976D2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Confidence colors */
    .confidence-high {
        color: #2E7D32;
        font-weight: 600;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: 600;
    }
    .confidence-low {
        color: #C62828;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E3F2FD;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'followup_answers' not in st.session_state:
    st.session_state.followup_answers = {}
if 'current_symptoms' not in st.session_state:
    st.session_state.current_symptoms = {}
if 'user_input_text' not in st.session_state:
    st.session_state.user_input_text = ""

# --- NLI VERIFICATION HELPERS (BATCHED) ---
def verify_claims_against_context(claims_list, context, api_key):
    """Verifies a batched list of claims against the MedlinePlus context in a single API call."""
    if not claims_list or not context or not api_key:
        return []
    
    try:
        genai.configure(api_key=api_key)
        
        # Updated to use gemini-2.5-flash-lite
        model = genai.GenerativeModel(
            'gemini-2.5-flash-lite', 
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        claims_str = json.dumps(claims_list)
        
        prompt = f"""
        You are a rigorous medical verification system.
        I will provide a JSON list of CLAIMS and a MedlinePlus medical CONTEXT.
        For each claim, verify if it is supported by the CONTEXT.

        CLAIMS:
        {claims_str}

        CONTEXT:
        {context}

        Respond STRICTLY with a JSON array containing objects with the following schema:
        [
            {{
                "statement": "string (MUST EXACTLY match the claim text provided)",
                "entailment_score": "High" | "Medium" | "Low",
                "context": "string (the supporting sentence from CONTEXT, or empty if not found)",
                "highlighted_context": "string (the exact substring to highlight, or empty if not found)"
            }}
        ]
        """
        
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings
        )
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        return json.loads(response_text.strip())
        
    except Exception as e:
        error_msg = str(e)
        print(f"NLI Verification Error Detail: {error_msg}")
        
        ui_msg = "API Error"
        if "429" in error_msg or "Quota" in error_msg:
            ui_msg = "Rate Limit Reached (Wait a minute before analyzing again)"
        elif "safety" in error_msg.lower() or "blocked" in error_msg.lower():
            ui_msg = "Blocked by API Safety Filters"
            
        return [{
            "statement": f"Verification Failed: {ui_msg}",
            "entailment_score": "Low",
            "context": "Check the terminal console for the exact error trace.",
            "highlighted_context": ""
        }]

def enrich_with_nli(result, api_key):
    """Batches symptoms and explanations into a SINGLE API call to prevent rate limiting."""
    
    rag_context = "\n".join(filter(None, [
        result.get('disease_description', ''),
        result.get('treatment_plan', ''),
        result.get('medications', ''),
        result.get('prevention', '')
    ]))
    
    # 1. Prepare Symptom Claims
    # Formatting as a proposition to help the NLI model understand the task
    symptoms = result.get('symptoms', [])
    symptom_claims = [f"The condition causes {s.replace('_', ' ').title()}." for s in symptoms] if symptoms else []
    
    # 2. Prepare Explanation Claims
    explanation_str = result.get('rag_explanation', result.get('explanation', ''))
    explanation_claims = [s.strip() + "." for s in explanation_str.replace('\n', ' ').split('.') if len(s.strip()) > 5] if explanation_str else []
    
    # If there is no MedlinePlus Context, bypass API entirely to save requests
    if not rag_context.strip():
        result['symptoms_nli'] = [{"statement": c, "entailment_score": "Medium", "context": "No MedlinePlus context available.", "highlighted_context": ""} for c in symptom_claims]
        result['explanation_nli'] = [{"statement": c, "entailment_score": "Medium", "context": "No MedlinePlus context available.", "highlighted_context": ""} for c in explanation_claims]
        return result

    # Combine all claims for a single batched API request
    combined_claims = symptom_claims + explanation_claims
    
    if not combined_claims:
        result['symptoms_nli'] = []
        result['explanation_nli'] = []
        return result
        
    # --- MAKE EXACTLY 1 API CALL HERE ---
    combined_results = verify_claims_against_context(combined_claims, rag_context, api_key)
    
    # Handle API crash safely
    if len(combined_results) == 1 and combined_results[0].get("statement", "").startswith("Verification Failed"):
        result['symptoms_nli'] = combined_results
        result['explanation_nli'] = combined_results
        return result
        
    # Unpack the batched results robustly mapping them back to their source strings
    result_dict = {item.get('statement', '').strip(): item for item in combined_results}
    
    symptoms_nli = []
    for claim in symptom_claims:
        matched = result_dict.get(claim.strip())
        if matched:
            symptoms_nli.append(matched)
        else:
            symptoms_nli.append({"statement": claim, "entailment_score": "Low", "context": "Failed to map API response.", "highlighted_context": ""})
            
    explanation_nli = []
    for claim in explanation_claims:
        matched = result_dict.get(claim.strip())
        if matched:
            explanation_nli.append(matched)
        else:
            explanation_nli.append({"statement": claim, "entailment_score": "Low", "context": "Failed to map API response.", "highlighted_context": ""})
            
    result['symptoms_nli'] = symptoms_nli
    result['explanation_nli'] = explanation_nli
        
    return result

def render_interactive_nli_ui(nli_list, full_context):
    """Renders a custom HTML/JS split-pane for interactive bidirectional highlighting."""
    if not nli_list:
        st.info("No statements to verify.")
        return

    # Convert Python data to JSON for JavaScript to consume
    nli_data_json = json.dumps(nli_list)
    
    # Safely escape quotes and newlines for HTML embedding
    safe_context = full_context.replace('"', '&quot;').replace("'", '&apos;').replace('\n', '<br><br>')

    # Custom HTML, CSS, and JS for the interactive UI
    html_code = f"""
    <style>
        .nli-container {{
            display: flex;
            gap: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-top: 10px;
        }}
        .pane {{
            flex: 1;
            padding: 15px;
            background: #ffffff;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            height: 450px;
            overflow-y: auto;
            line-height: 1.6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .pane-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #1565c0;
            margin-bottom: 15px;
            border-bottom: 2px solid #bbdefb;
            padding-bottom: 5px;
            position: sticky;
            top: 0;
            background: #ffffff;
            z-index: 10;
        }}
        .claim {{
            cursor: pointer;
            padding: 4px 6px;
            border-radius: 4px;
            transition: background-color 0.2s;
            display: inline;
            margin-right: 4px;
        }}
        .claim:hover {{
            background-color: #e3f2fd;
        }}
        .claim.active {{
            background-color: #bbdefb;
            font-weight: 500;
            border-bottom: 2px solid #1976d2;
        }}
        .claim-high::before {{ content: "✅ "; font-size: 0.9em; }}
        .claim-medium::before {{ content: "⚠️ "; font-size: 0.9em; }}
        .claim-low::before {{ content: "❌ "; font-size: 0.9em; }}
        
        .highlighted-source {{
            background-color: #c8e6c9;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
            transition: all 0.3s ease;
        }}
        .instruction {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 10px;
            font-style: italic;
        }}
    </style>

    <div class="instruction">👆 Click any sentence on the left to locate its supporting evidence on the right.</div>
    
    <div class="nli-container">
        <div class="pane" id="hypothesis-pane">
            <div class="pane-title">AI Generated Claims (Hypothesis)</div>
            <div id="claims-container"></div>
        </div>
        
        <div class="pane" id="premise-pane">
            <div class="pane-title">MedlinePlus Reference (Premise)</div>
            <div id="context-container">{safe_context}</div>
        </div>
    </div>

    <script>
        const nliData = {nli_data_json};
        const claimsContainer = document.getElementById('claims-container');
        const contextContainer = document.getElementById('context-container');
        const originalContextHTML = contextContainer.innerHTML;

        // Render claims
        nliData.forEach((item, index) => {{
            const span = document.createElement('span');
            span.className = 'claim claim-' + item.entailment_score.toLowerCase();
            span.innerText = item.statement + ' ';
            
            span.onclick = function() {{
                // Remove active class from all claims
                document.querySelectorAll('.claim').forEach(el => el.classList.remove('active'));
                // Add active class to clicked claim
                span.classList.add('active');
                
                // Reset context pane to original text
                contextContainer.innerHTML = originalContextHTML;
                
                // Highlight the specific text if it exists
                if ((item.entailment_score === 'High' || item.entailment_score === 'Medium') && item.highlighted_context) {{
                    const highlightText = item.highlighted_context;
                    // Safely escape regex characters
                    const escapedText = highlightText.replace(/[-\\/\\\\^$*+?.()|[\\]{{}}]/g, '\\\\$&');
                    const regex = new RegExp('(' + escapedText + ')', 'gi');
                    
                    contextContainer.innerHTML = contextContainer.innerHTML.replace(
                        regex, 
                        '<span class="highlighted-source" id="scroll-target">$1</span>'
                    );
                    
                    // Scroll the highlighted source into view
                    const highlightedEl = document.getElementById('scroll-target');
                    if(highlightedEl) {{
                        highlightedEl.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    }}
                }} else if (item.entailment_score === 'Low') {{
                    contextContainer.innerHTML = "<div style='color: #c62828; padding: 10px; background: #ffcdd2; border-radius: 5px; margin-bottom: 10px;'>No supporting evidence found in MedlinePlus context. AI may be hallucinating this claim.</div>" + originalContextHTML;
                }}
            }};
            
            claimsContainer.appendChild(span);
        }});
    </script>
    """
    
    # Render the custom HTML component in Streamlit
    components.html(html_code, height=520, scrolling=False)
# --- END NLI HELPERS ---

def initialize_chatbot():
    """Initialize the hybrid chatbot."""
    api_key = os.getenv('GEMINI_API_KEY')
    model_path = 'models/disease_bayesian_network.pkl'
    
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found in environment variables!")
        st.info("Please set your API key in the `.env` file or sidebar.")
        return None
    
    try:
        chatbot = HybridMedicalChatbot(model_path, api_key)
        return chatbot
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please train the model first by running `03_bayesian_network_binary.ipynb`")
        return None
    except Exception as e:
        st.error(f"⚠️ Error initializing chatbot: {e}")
        return None

def create_probability_chart(predictions):
    """Create an interactive bar chart for predictions."""
    diseases = [p[0] for p in predictions]
    probabilities = [p[1] * 100 for p in predictions]
    
    # Color code by confidence
    colors = []
    for prob in probabilities:
        if prob > 75:
            colors.append('#2e7d32')  # Green
        elif prob > 50:
            colors.append('#f57c00')  # Orange
        else:
            colors.append('#1976d2')  # Blue
    
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities,
            y=diseases,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Disease Probability Distribution",
        xaxis_title="Probability (%)",
        yaxis_title="Disease",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def display_symptoms(symptoms):
    """Display symptoms as chips."""
    if symptoms:
        st.markdown("**Identified Symptoms:**")
        chips_html = "".join([
            f'<span class="symptom-chip">{s.replace("_", " ").title()}</span>'
            for s in symptoms
        ])
        st.markdown(chips_html, unsafe_allow_html=True)

def display_predictions(predictions):
    """Display predictions in a formatted table."""
    if predictions:
        df = pd.DataFrame(predictions, columns=['Disease', 'Probability'])
        df['Probability'] = df['Probability'].apply(lambda x: f'{x*100:.2f}%')
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'Disease', 'Probability']]
        
        st.dataframe(
            df,
            hide_index=True,
            width='stretch',
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Disease": st.column_config.TextColumn("Disease", width="large"),
                "Probability": st.column_config.TextColumn("Confidence", width="medium"),
            }
        )

# Main UI
st.markdown('<div class="main-header">🏥 Medical Diagnosis Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Disease Prediction with Zero Hallucinations</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input (optional override)
    api_key_input = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        help="Leave empty to use .env file",
        placeholder="AIzaSy..."
    )
    
    if api_key_input:
        os.environ['GEMINI_API_KEY'] = api_key_input
    
    st.divider()
    
    # About section
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **Bayesian Network** for accurate, hallucination-free predictions
    - **Gemini API** for natural language understanding
    - **131 symptoms** and **41 diseases**
    
    **How it works:**
    1. Describe your symptoms naturally
    2. AI extracts symptoms from your text
    3. Bayesian Network predicts diseases
    4. Get results with confidence scores
    """)
    
    st.divider()
    
    # Statistics
    st.header("📊 Statistics")
    if st.session_state.chatbot:
        st.metric("Diseases", len(st.session_state.chatbot.diseases))
        st.metric("Symptoms", len(st.session_state.chatbot.symptom_cols))
        st.metric("Diagnoses Made", len(st.session_state.diagnosis_history))
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear History", width='stretch'):
        st.session_state.diagnosis_history = []
        st.session_state.current_result = None
        st.rerun()

# Initialize chatbot
if st.session_state.chatbot is None:
    with st.spinner("Initializing chatbot..."):
        st.session_state.chatbot = initialize_chatbot()

# Main content
if st.session_state.chatbot:
    # Input section with better organization
    st.markdown('<div class="section-header">💬 Step 1: Describe Your Symptoms</div>', unsafe_allow_html=True)
    
    # Help text
    st.info("📝 **How to use:** Describe your symptoms in plain English. The AI will automatically extract and analyze them.")
    
    # Example symptoms in an expander
    with st.expander("💡 See Example Symptoms"):
        st.markdown("""
        **Example 1:** "I have a high fever, severe headache, and I've been vomiting. I also have chills and muscle pain."
        
        **Example 2:** "I'm experiencing increased thirst, frequent urination, and constant fatigue."
        
        **Example 3:** "I have a persistent cough, runny nose, and sore throat for the past 3 days."
        """)
    
    # Input area
    user_input = st.text_area(
        "Enter your symptoms here:",
        placeholder="Example: I have a high fever, severe headache, and I've been vomiting...",
        height=120,
        key="symptom_input",
        help="Be as specific as possible about your symptoms"
    )
    
    # Diagnose button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        diagnose_button = st.button(
            "🔍 Analyze Symptoms",
            type="primary",
            width='stretch',
            disabled=not user_input.strip()
        )
    
    st.divider()
    
    # Process diagnosis
    if diagnose_button and user_input.strip():
        if user_input.strip():
            with st.status("🩺 Processing Medical Diagnosis...", expanded=True) as status:
                try:
                    st.write("Analyzing symptoms and predicting diseases...")
                    result = st.session_state.chatbot.diagnose(user_input)
                    
                    st.write("Performing Batched Context NLI Verification...")
                    api_key = os.getenv('GEMINI_API_KEY')
                    result = enrich_with_nli(result, api_key)
                    
                    st.session_state.current_result = result
                    
                    # Store current symptoms for follow-up questions
                    if result.get('symptoms'):
                        st.session_state.current_symptoms = {s: 1 for s in result['symptoms']}
                    
                    # Clear previous follow-up answers
                    st.session_state.followup_answers = {}
                    
                    st.session_state.diagnosis_history.append({
                        'input': user_input,
                        'result': result
                    })
                    status.update(label="Diagnosis Complete!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Diagnosis Failed", state="error", expanded=False)
                    st.error(f"Error during diagnosis: {e}")
        else:
            st.warning("Please describe your symptoms first!")
    
    # Display results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        st.divider()
        st.header("📋 Diagnosis Results")
        
        # Symptoms section
        if result['symptoms']:
            display_symptoms(result['symptoms'])
            st.markdown("")
        else:
            st.warning("⚠️ No symptoms were identified from your description. Please be more specific.")
        
        # SAFE RESPONSE STRATEGY: Always show predictions with appropriate warnings
        # Check if this is a low confidence warning case
        if result.get('low_confidence_warning'):
            # Very low confidence - show predictions with STRONG warnings
            st.error("""
            ⚠️ **CRITICAL: LOW CONFIDENCE PREDICTION**
            
            The predictions below have very low confidence and may not be accurate.
            **MEDICAL CONSULTATION IS REQUIRED** - Do not rely on these predictions alone.
            """)
            
            # Show the explanation with warnings
            st.markdown(result.get('explanation', ''))
            
            # Show predictions in a warning box
            if result.get('predictions'):
                st.divider()
                st.subheader("⚠️ Low Confidence Possibilities")
                st.caption("These are statistical possibilities only - NOT a diagnosis")
                
                # Show top 3 predictions with warning styling
                for i, (disease, prob) in enumerate(result['predictions'][:3], 1):
                    st.warning(f"**{i}. {disease}** - {prob*100:.1f}% probability (LOW CONFIDENCE)")
            
            # Show RAG information if available
            if 'treatment_plan' in result:
                st.divider()
                st.info("📚 **Additional Medical Information** (for reference only - consult a doctor)")
                
                # Create tabs for RAG information
                tab1, tab2, tab3, tab4, tab_nli = st.tabs([
                    "💊 Treatment Info",
                    "💉 Medications",
                    "🔔 Next Steps",
                    "🛡️ Prevention",
                    "🔬 Context NLI"
                ])
                
                with tab1:
                    st.markdown("### Treatment Information")
                    st.caption("⚠️ This information is for educational purposes. Consult a healthcare professional.")
                    if 'treatment_plan' in result:
                        st.markdown(result['treatment_plan'])
                    else:
                        st.info("Treatment information not available.")
                
                with tab2:
                    st.markdown("### Medication Information")
                    st.caption("⚠️ NEVER self-medicate. All medications must be prescribed by a doctor.")
                    if 'medications' in result:
                        st.markdown(result['medications'])
                    else:
                        st.info("Medication information not available.")
                
                with tab3:
                    st.markdown("### Recommended Next Steps")
                    if 'next_steps' in result:
                        st.markdown(result['next_steps'])
                    else:
                        st.info("Please consult a healthcare professional for guidance.")
                
                with tab4:
                    st.markdown("### Prevention & Risk Reduction")
                    if 'prevention' in result:
                        st.markdown(result['prevention'])
                    else:
                        st.info("Prevention information not available.")
                        
                with tab_nli:
                    st.subheader("🔬 Interactive Context Verification")
                    
                    # Rebuild context string
                    rag_context_str = "\n".join(filter(None, [
                        result.get('disease_description', ''),
                        result.get('treatment_plan', ''),
                        result.get('medications', ''),
                        result.get('prevention', '')
                    ]))
                    
                    # Combine symptoms and explanation into one interactive view
                    combined_nli = result.get('symptoms_nli', []) + result.get('explanation_nli', [])
                    render_interactive_nli_ui(combined_nli, rag_context_str)
                
                # Show source
                if 'source_url' in result:
                    st.caption(f"**Source:** [{result.get('source_name', 'MedlinePlus')}]({result['source_url']})")
            
            # Show follow-up questions as optional (if any)
            if result.get('followup_questions'):
                st.divider()
                st.markdown("### 🔍 Optional: Answer Questions to Refine Prediction")
                st.caption("*Note: Even with more information, medical consultation is still required*")
                
                # Display questions
                for i, q in enumerate(result['followup_questions']):
                    symptom = q['symptom']
                    question = q['question']
                    
                    st.markdown(f"**{i+1}. {question}**")
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    
                    with col1:
                        if st.button("✅ Yes", key=f"yes_{symptom}_low"):
                            st.session_state.followup_answers[symptom] = 1
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ No", key=f"no_{symptom}_low"):
                            st.session_state.followup_answers[symptom] = 0
                            st.rerun()
                    
                    with col3:
                        if symptom in st.session_state.followup_answers:
                            answer = "Yes" if st.session_state.followup_answers[symptom] == 1 else "No"
                            st.markdown(f"*Answer: {answer}*")
                
                # Update button
                if st.session_state.followup_answers:
                    st.divider()
                    if st.button("🔄 Update Prediction with Answers", type="primary", use_container_width=True, key="update_low_conf"):
                        with st.status("🔄 Updating prediction...", expanded=True) as status:
                            try:
                                st.write("Re-evaluating symptom data...")
                                all_symptoms = st.session_state.current_symptoms.copy()
                                all_symptoms.update(st.session_state.followup_answers)
                                
                                predictions = st.session_state.chatbot.predict_disease(all_symptoms, top_n=5)
                                
                                if predictions and len(predictions) > 0:
                                    symptom_list = [s for s, v in all_symptoms.items() if v == 1]
                                    explanation = st.session_state.chatbot.explain_results(predictions, symptom_list)
                                    new_confidence = predictions[0][1]
                                    
                                    updated_result = {
                                        'symptoms': symptom_list,
                                        'predictions': predictions,
                                        'explanation': explanation,
                                        'requires_followup': new_confidence < st.session_state.chatbot.FOLLOWUP_THRESHOLD,
                                        'followup_questions': [],
                                        'confidence': new_confidence,
                                        'low_confidence_warning': new_confidence < st.session_state.chatbot.MINIMUM_CONFIDENCE
                                    }
                                    
                                    st.write("Performing Batched Context NLI Verification...")
                                    api_key = os.getenv('GEMINI_API_KEY')
                                    updated_result = enrich_with_nli(updated_result, api_key)
                                    
                                    st.success(f"✅ Prediction updated! Confidence: {new_confidence*100:.1f}%")
                                    
                                    st.session_state.current_result = updated_result
                                    st.session_state.followup_answers = {}
                                    status.update(label="Update Complete", state="complete", expanded=False)
                                    st.rerun()
                                else:
                                    st.warning("⚠️ Unable to Generate Prediction. Please consult a professional.")
                                    status.update(label="Failed to predict", state="error", expanded=False)
                            except Exception as e:
                                st.error(f"Error updating prediction: {str(e)}")
                                status.update(label="Error processing", state="error", expanded=False)
            
            # Medical disclaimer
            st.divider()
            st.error("""
            🚨 **MANDATORY MEDICAL CONSULTATION REQUIRED**
            
            This is an AI-based educational tool with LOW CONFIDENCE in this prediction. 
            **You MUST consult a qualified healthcare provider** for proper diagnosis and treatment.
            Do NOT make any medical decisions based on this information alone.
            """)
        
        # Normal predictions with good confidence
        elif result.get('predictions') and len(result['predictions']) > 0:
            # Display identified symptoms
            display_symptoms(result['symptoms'])
            
            # Results section header
            st.markdown('<div class="section-header">📋 Step 2: Analysis Results</div>', unsafe_allow_html=True)
            
            # Top prediction card with better styling
            top_disease = result['predictions'][0][0]
            top_prob = result['predictions'][0][1]
            
            if top_prob > 0.75:
                confidence_class = "confidence-high"
                confidence_emoji = "✅"
                confidence_text = "High Confidence"
                confidence_color = "#2E7D32"
            elif top_prob > 0.50:
                confidence_class = "confidence-medium"
                confidence_emoji = "⚠️"
                confidence_text = "Medium Confidence"
                confidence_color = "#F57C00"
            else:
                confidence_class = "confidence-low"
                confidence_emoji = "❌"
                confidence_text = "Low Confidence"
                confidence_color = "#C62828"
            
            # Enhanced top prediction card
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                padding: 1.5rem;
                border-radius: 12px;
                margin: 1rem 0;
                border-left: 5px solid {confidence_color};
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            ">
                <h3 style="margin: 0 0 0.5rem 0; color: #1565C0;">
                    {confidence_emoji} Top Prediction: {top_disease}
                </h3>
                <p style="margin: 0; font-size: 1.1rem;">
                    <span style="color: {confidence_color}; font-weight: 600;">
                        Confidence: {top_prob*100:.1f}%
                    </span>
                    <span style="color: #666; margin-left: 1rem;">
                        ({confidence_text})
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced NLI Verification Section
            if 'nli_verification' in result:
                st.markdown("---")
                
                # Prominent header with icon
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin: 1rem 0;
                    border-left: 5px solid #4CAF50;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                ">
                    <h3 style="margin: 0; color: #2E7D32;">
                        🔍 Information Verification & Accuracy Check
                    </h3>
                    <p style="margin: 0.5rem 0 0 0; color: #555; font-size: 0.95rem;">
                        AI-powered verification using Natural Language Inference (NLI)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                verification = result['nli_verification']
                
                # Overall verification score with progress bar
                verification_score = verification['verification_score']
                score_percentage = int(verification_score * 100)
                
                # Color based on score
                if verification_score >= 0.67:
                    score_color = "#4CAF50"  # Green
                    score_emoji = "✅"
                    score_label = "HIGH CONFIDENCE"
                elif verification_score >= 0.33:
                    score_color = "#FF9800"  # Orange
                    score_emoji = "⚠️"
                    score_label = "MEDIUM CONFIDENCE"
                else:
                    score_color = "#F44336"  # Red
                    score_emoji = "❌"
                    score_label = "LOW CONFIDENCE"
                
                # Display overall score
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 1.2rem;
                    border-radius: 10px;
                    margin: 1rem 0;
                    border: 2px solid {score_color};
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-size: 1.1rem; font-weight: 600; color: #333;">
                            {score_emoji} Overall Verification Score
                        </span>
                        <span style="font-size: 1.3rem; font-weight: 700; color: {score_color};">
                            {score_percentage}%
                        </span>
                    </div>
                    <div style="
                        background: #f0f0f0;
                        border-radius: 10px;
                        height: 20px;
                        overflow: hidden;
                    ">
                        <div style="
                            background: linear-gradient(90deg, {score_color}, {score_color}dd);
                            width: {score_percentage}%;
                            height: 100%;
                            border-radius: 10px;
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                    <div style="margin-top: 0.5rem; text-align: center;">
                        <span style="
                            display: inline-block;
                            background: {score_color}22;
                            color: {score_color};
                            padding: 0.3rem 1rem;
                            border-radius: 15px;
                            font-weight: 600;
                            font-size: 0.85rem;
                        ">
                            {score_label}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Summary message
                if verification['overall_verified']:
                    st.success(f"✅ **{verification['summary']}**")
                else:
                    st.warning(f"⚠️ **{verification['summary']}**")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Detailed verification checks with enhanced cards
                st.markdown("#### 📋 Detailed Verification Checks")
                
                # Detailed checks in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    symptom_check = verification['symptom_match']
                    
                    # Determine card styling
                    if symptom_check['verified']:
                        card_bg = "#E8F5E9"
                        card_border = "#4CAF50"
                        status_text = "✅ Verified"
                        status_color = "#2E7D32"
                    else:
                        card_bg = "#FFF3E0"
                        card_border = "#FF9800"
                        status_text = "⚠️ Review Needed"
                        status_color = "#E65100"
                    
                    st.markdown(f"""
                    <div style="
                        background: {card_bg};
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 4px solid {card_border};
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    ">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🩺</div>
                        <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">Symptom Match</div>
                        <div style="
                            font-weight: 700;
                            color: {status_color};
                            font-size: 0.95rem;
                            margin-bottom: 0.5rem;
                        ">
                            {status_text}
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.85rem;
                            color: #666;
                            margin-bottom: 0.5rem;
                        ">
                            Confidence: {int(symptom_check['confidence']*100)}%
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.8rem;
                            color: #666;
                        ">
                            <strong>Status:</strong> {symptom_check['label']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📖 View Reasoning"):
                        st.write(symptom_check['reasoning'])
                
                with col2:
                    treatment_check = verification['treatment_check']
                    
                    # Determine card styling
                    if treatment_check['verified']:
                        card_bg = "#E8F5E9"
                        card_border = "#4CAF50"
                        status_text = "✅ Appropriate"
                        status_color = "#2E7D32"
                    else:
                        card_bg = "#FFF3E0"
                        card_border = "#FF9800"
                        status_text = "⚠️ Review Needed"
                        status_color = "#E65100"
                    
                    st.markdown(f"""
                    <div style="
                        background: {card_bg};
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 4px solid {card_border};
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    ">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💊</div>
                        <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">Treatment Plan</div>
                        <div style="
                            font-weight: 700;
                            color: {status_color};
                            font-size: 0.95rem;
                            margin-bottom: 0.5rem;
                        ">
                            {status_text}
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.85rem;
                            color: #666;
                            margin-bottom: 0.5rem;
                        ">
                            Confidence: {int(treatment_check['confidence']*100)}%
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.8rem;
                            color: #666;
                        ">
                            <strong>Status:</strong> {treatment_check['label']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📖 View Reasoning"):
                        st.write(treatment_check['reasoning'])
                
                with col3:
                    medication_check = verification['medication_check']
                    
                    # Determine card styling
                    if medication_check['verified']:
                        card_bg = "#E8F5E9"
                        card_border = "#4CAF50"
                        status_text = "✅ Safe"
                        status_color = "#2E7D32"
                    else:
                        card_bg = "#FFF3E0"
                        card_border = "#FF9800"
                        status_text = "⚠️ Consult Doctor"
                        status_color = "#E65100"
                    
                    st.markdown(f"""
                    <div style="
                        background: {card_bg};
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 4px solid {card_border};
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    ">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💉</div>
                        <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">Medication Safety</div>
                        <div style="
                            font-weight: 700;
                            color: {status_color};
                            font-size: 0.95rem;
                            margin-bottom: 0.5rem;
                        ">
                            {status_text}
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.85rem;
                            color: #666;
                            margin-bottom: 0.5rem;
                        ">
                            Confidence: {int(medication_check['confidence']*100)}%
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.8rem;
                            color: #666;
                        ">
                            <strong>Status:</strong> {medication_check['label']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📖 View Reasoning"):
                        st.write(medication_check['reasoning'])
                
                # Show contradictions if any
                if verification.get('contradictions'):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.error("🚨 **Contradictions Detected in Medical Information**")
                    for i, contradiction in enumerate(verification['contradictions'], 1):
                        st.markdown(f"""
                        <div style="
                            background: #FFEBEE;
                            padding: 1rem;
                            border-radius: 8px;
                            border-left: 4px solid #F44336;
                            margin: 0.5rem 0;
                        ">
                            <strong>Contradiction {i}:</strong> {contradiction['type']}<br>
                            <span style="color: #666; font-size: 0.9rem;">{contradiction['details']}</span><br>
                            <span style="color: #999; font-size: 0.85rem;">Confidence: {int(contradiction.get('confidence', 0)*100)}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Information box about NLI
                st.info("""
                **ℹ️ About NLI Verification:**
                
                Natural Language Inference (NLI) is an AI technique that verifies the logical consistency between:
                - Your symptoms and the predicted disease description
                - The disease and recommended treatments
                - The condition and suggested medications
                
                This helps ensure the information provided is accurate and consistent with medical knowledge.
                """)
                
                st.markdown("---")
            
            # Tabs for different views
            # Check if RAG information is available
            has_rag = 'treatment_plan' in result
            
            st.markdown("### 📑 Detailed Information")
            
            if has_rag:
                tab1, tab2, tab_nli, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 All Predictions", 
                    "📖 Explanation", 
                    "🔬 Context NLI",
                    "💊 Treatment",
                    "💉 Medications",
                    "🔔 Next Steps",
                    "🛡️ Prevention"
                ])
            else:
                tab1, tab2, tab3, tab_nli = st.tabs(["📊 Predictions", "📋 Details", "💬 Explanation", "🔬 Context NLI"])
            
            with tab1:
                st.markdown("#### Probability Distribution")
                fig = create_probability_chart(result['predictions'])
                st.plotly_chart(fig, width='stretch')
                
                st.markdown("#### All Predictions")
                display_predictions(result['predictions'])
            
            with tab2:
                st.subheader("AI Explanation")
                
                # Show RAG explanation if available, otherwise Gemini explanation
                if has_rag and 'rag_explanation' in result:
                    st.markdown(result['rag_explanation'])
                    
                    # Show disease description
                    if 'disease_description' in result:
                        st.divider()
                        st.markdown("**Disease Overview:**")
                        st.info(result['disease_description'])
                    
                    # Show source citation
                    if 'source_url' in result:
                        st.divider()
                        st.markdown(f"**Source:** [{result.get('source_name', 'MedlinePlus')}]({result['source_url']})")
                else:
                    st.info(result['explanation'])

            with tab_nli:
                st.subheader("🔬 Interactive Context Verification")
                
                # Rebuild context string
                rag_context_str = "\n".join(filter(None, [
                    result.get('disease_description', ''),
                    result.get('treatment_plan', ''),
                    result.get('medications', ''),
                    result.get('prevention', '')
                ]))
                
                # Combine symptoms and explanation into one interactive view
                combined_nli = result.get('symptoms_nli', []) + result.get('explanation_nli', [])
                render_interactive_nli_ui(combined_nli, rag_context_str)
            
            # RAG-specific tabs
            if has_rag:
                with tab3:
                    st.subheader("💊 Treatment Plan")
                    if 'treatment_plan' in result:
                        st.markdown(result['treatment_plan'])
                    else:
                        st.info("Treatment information not available. Please consult a healthcare professional.")
                
                with tab4:
                    st.subheader("💉 Medication Information")
                    if 'medications' in result:
                        st.markdown(result['medications'])
                        
                        # Show complications if available
                        if 'complications' in result and result['complications']:
                            st.divider()
                            st.warning("**Potential Complications:**")
                            st.markdown(result['complications'])
                    else:
                        st.info("Medication information not available. Please consult a healthcare professional.")
                
                with tab5:
                    st.subheader("🔔 Next Steps")
                    if 'next_steps' in result:
                        st.markdown(result['next_steps'])
                    else:
                        st.info("Please consult a healthcare professional for guidance on next steps.")
                
                with tab6:
                    st.subheader("🛡️ Prevention & Risk Reduction")
                    if 'prevention' in result:
                        st.markdown(result['prevention'])
                    else:
                        st.info("Prevention information not available.")
                    
                    # Show source
                    if 'source_url' in result:
                        st.divider()
                        st.caption(f"**Source:** [{result.get('source_name', 'MedlinePlus')}]({result['source_url']})")
            else:
                # Fallback tabs without RAG
                with tab3:
                    st.markdown("### AI Explanation")
                    st.info(result['explanation'])

            
            # Follow-up questions section (for medium confidence predictions)
            if result.get('requires_followup') and result.get('followup_questions'):
                st.divider()
                st.markdown("### 🔍 Improve Prediction Accuracy")
                
                # Show info based on confidence level
                if top_prob < 0.50:
                    st.warning(f"""
                    ⚠️ **Medium-Low Confidence ({top_prob*100:.1f}%)**
                    
                    The current prediction has medium-low confidence. Answering the questions below 
                    can help improve the accuracy of the diagnosis.
                    """)
                else:
                    st.info(f"""
                    📊 **Confidence: {top_prob*100:.1f}%**
                    
                    You can optionally answer these questions to further refine the prediction.
                    """)
                
                # Display follow-up questions
                for i, q in enumerate(result['followup_questions']):
                    symptom = q['symptom']
                    question = q['question']
                    
                    st.markdown(f"**{i+1}. {question}**")
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    
                    with col1:
                        if st.button("✅ Yes", key=f"yes_{symptom}_normal"):
                            st.session_state.followup_answers[symptom] = 1
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ No", key=f"no_{symptom}_normal"):
                            st.session_state.followup_answers[symptom] = 0
                            st.rerun()
                    
                    with col3:
                        if symptom in st.session_state.followup_answers:
                            answer = "Yes" if st.session_state.followup_answers[symptom] == 1 else "No"
                            st.markdown(f"*Answer: {answer}*")
                    
                    st.markdown("")
                
                # Update prediction button
                if st.session_state.followup_answers:
                    st.divider()
                    if st.button("🔄 Update Prediction with Answers", type="primary", width='stretch', key="update_normal_conf"):
                        with st.status("🔄 Updating prediction...", expanded=True) as status:
                            try:
                                st.write("Re-evaluating symptoms with new answers...")
                                # Combine original symptoms with follow-up answers
                                all_symptoms = st.session_state.current_symptoms.copy()
                                all_symptoms.update(st.session_state.followup_answers)
                                
                                # Make new prediction
                                predictions = st.session_state.chatbot.predict_disease(all_symptoms, top_n=5)
                                
                                if predictions and len(predictions) > 0:
                                    # Get explanation
                                    symptom_list = [s for s, v in all_symptoms.items() if v == 1]
                                    explanation = st.session_state.chatbot.explain_results(predictions, symptom_list)
                                    
                                    # Check new confidence
                                    new_confidence = predictions[0][1]
                                    top_disease = predictions[0][0]
                                    
                                    # Get RAG information if available
                                    rag_info = None
                                    if st.session_state.chatbot.use_rag and hasattr(st.session_state.chatbot, 'rag') and st.session_state.chatbot.rag:
                                        try:
                                            rag_info = st.session_state.chatbot.rag.explain_diagnosis(
                                                disease=top_disease,
                                                confidence=new_confidence,
                                                symptoms=symptom_list
                                            )
                                        except Exception as e:
                                            st.warning(f"Could not retrieve detailed medical information: {e}")
                                    
                                    # Create updated result
                                    updated_result = {
                                        'symptoms': symptom_list,
                                        'predictions': predictions,
                                        'explanation': explanation,
                                        'requires_followup': new_confidence < st.session_state.chatbot.FOLLOWUP_THRESHOLD,
                                        'followup_questions': [],
                                        'confidence': new_confidence,
                                        'low_confidence_warning': new_confidence < st.session_state.chatbot.MINIMUM_CONFIDENCE
                                    }
                                    
                                    # Add RAG information if available
                                    if rag_info:
                                        updated_result.update({
                                            'rag_explanation': rag_info['explanation'],
                                            'treatment_plan': rag_info['treatment_plan'],
                                            'medications': rag_info['medications'],
                                            'next_steps': rag_info['next_steps'],
                                            'prevention': rag_info['prevention'],
                                            'source_url': rag_info['source_url'],
                                            'source_name': rag_info['source_name'],
                                            'disease_description': rag_info['disease_description'],
                                            'complications': rag_info['complications']
                                        })
                                        
                                        # Add NLI verification if available
                                        if 'nli_verification' in rag_info:
                                            updated_result['nli_verification'] = rag_info['nli_verification']
                                    
                                    st.write("Performing Batched Context NLI Verification...")
                                    api_key = os.getenv('GEMINI_API_KEY')
                                    updated_result = enrich_with_nli(updated_result, api_key)
                                    
                                    st.success(f"✅ Prediction updated! New confidence: {new_confidence*100:.1f}%")
                                    
                                    st.session_state.current_result = updated_result
                                    st.session_state.current_symptoms = all_symptoms
                                    st.session_state.followup_answers = {}
                                    status.update(label="Update Complete", state="complete", expanded=False)
                                    st.rerun()
                                else:
                                    st.warning("""
                                    ⚠️ **Unable to Generate Prediction**
                                    
                                    The symptom combination is unusual. Please consult a healthcare professional for proper diagnosis.
                                    
                                    **Recommended Actions:**
                                    - 🏥 Schedule an appointment with a doctor
                                    - 📝 Bring your symptom list
                                    - 🔍 Get proper medical tests
                                    """)
                                    status.update(label="Prediction Failed", state="error", expanded=False)
                            
                            except Exception as e:
                                st.error(f"Error updating prediction: {str(e)}")
                                status.update(label="Error processing", state="error", expanded=False)
            
            # Medical disclaimer
            st.divider()
            st.warning("""
            ⚠️ **MEDICAL DISCLAIMER**
            
            This is an AI-based educational tool and NOT a substitute for professional medical advice, 
            diagnosis, or treatment. Always consult a qualified healthcare provider for any medical 
            concerns or before making health-related decisions.
            """)
        
        else:
            # No predictions available
            st.error("⚠️ **Unable to Make Prediction**")
            st.warning("""
            This symptom combination is not well-represented in the training data. 
            The model cannot make a reliable prediction for these symptoms.
            
            **Please consult a healthcare professional for proper diagnosis.**
            """)
            
            # Still show the explanation if available
            if result.get('explanation'):
                st.info(result['explanation'])
    
    
    # History section
    if st.session_state.diagnosis_history:
        st.divider()
        st.header("📜 Diagnosis History")
        
        with st.expander(f"View {len(st.session_state.diagnosis_history)} previous diagnoses"):
            for i, item in enumerate(reversed(st.session_state.diagnosis_history), 1):
                st.markdown(f"**#{len(st.session_state.diagnosis_history) - i + 1}:** {item['input'][:100]}...")
                if item['result']['predictions']:
                    top = item['result']['predictions'][0]
                    st.markdown(f"→ **{top[0]}** ({top[1]*100:.1f}%)")
                st.markdown("---")

else:
    # Chatbot initialization failed
    st.error("Failed to initialize chatbot. Please check the configuration.")
    st.info("""
    **Setup Instructions:**
    1. Create a `.env` file with your Gemini API key
    2. Train the model by running `03_bayesian_network_binary.ipynb`
    3. Restart this app
    
    See `HYBRID_CHATBOT_SETUP.md` for detailed instructions.
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Powered by Bayesian Networks + Google Gemini AI | 
    <a href="https://github.com" target="_blank">GitHub</a> | 
    Built with Streamlit
</div>
""", unsafe_allow_html=True)