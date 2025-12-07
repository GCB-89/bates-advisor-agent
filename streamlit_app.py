"""
Streamlit Chat UI for Bates Technical College Student Advisor
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from dotenv import load_dotenv
import html

# Load environment variables from .env file
load_dotenv()

from src.orchestrator import MultiAgentOrchestrator

# Page configuration
st.set_page_config(
    page_title="Bates Tech Student Advisor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for Bates Tech branding
st.markdown("""
<style>
    /* Main color scheme - Bates Tech Navy Blue and Gold */
    :root {
        --bates-navy: #1a3a5c;
        --bates-dark-navy: #0d2840;
        --bates-gold: #d4a849;
        --bates-light-gold: #e8c36a;
    }
    
    /* Main content area - dark background */
    .main {
        background-color: #0d1117 !important;
    }
    
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
        background-color: #0d1117;
    }
    
    /* Make all main text white */
    .main, .main p, .main span, .main div {
        color: #e6e6e6 !important;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        color: #1a3a5c;
        font-size: 2.2rem;
        font-weight: bold;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 1rem;
    }
    
    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #1a3a5c 0%, #2a5a8c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .welcome-card h2 {
        color: #d4a849;
        margin-bottom: 0.5rem;
    }
    
    .welcome-card p {
        color: #e0e0e0;
        margin: 0;
    }
    
    /* Chat container */
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        min-height: 400px;
        border: 1px solid #e0e0e0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #0d2840 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent;
        border: 1px solid #d4a849;
        color: #d4a849 !important;
        transition: all 0.3s ease;
        font-size: 0.85rem;
        padding: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #d4a849;
        color: #1a3a5c !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #d4a849 !important;
    }
    
    /* Custom chat bubbles */
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 18px;
        margin-bottom: 12px;
        max-width: 80%;
        line-height: 1.5;
        font-size: 15px;
        word-wrap: break-word;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #5cb85c 0%, #4a9f4a 100%);
        color: white;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 4px;
        text-align: left;
    }
    
    .assistant-bubble {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        margin-left: 0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        text-align: left;
    }
    
    /* Hide default Streamlit chat styling */
    div[data-testid="stChatMessage"] {
        display: none !important;
    }
    
    /* Primary buttons */
    .stButton > button {
        background-color: #1a3a5c;
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #d4a849;
        color: #1a3a5c;
    }
    
    /* Chat input styling - visible border */
    .stChatInput {
        background-color: #1a3a5c !important;
        padding: 4px !important;
        border-radius: 30px !important;
    }
    
    .stChatInput > div {
        background-color: white !important;
        border-radius: 26px !important;
        border: none !important;
    }
    
    .stChatInput [data-testid="stChatInputContainer"] {
        background-color: white !important;
        border: none !important;
        border-radius: 26px !important;
    }
    
    .stChatInput textarea {
        background-color: white !important;
        color: #1a3a5c !important;
    }
    
    .stChatInput textarea::placeholder {
        color: #1a3a5c !important;
        opacity: 0.6 !important;
    }
    
    .stChatInput button {
        background-color: #d4a849 !important;
        border-radius: 50% !important;
    }
    
    .stChatInput button svg {
        fill: #1a3a5c !important;
    }
    
    .stChatInput textarea::placeholder {
        color: #1a3a5c !important;
        opacity: 0.7;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f4f8;
        color: #1a3a5c;
    }
    
    /* Divider */
    hr {
        border-color: #d4a849;
    }
    
    /* Link styling */
    a {
        color: #d4a849 !important;
    }
    
    /* Hide default header decoration */
    .stDeployButton {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Header with welcome card
st.markdown("""
<div class="welcome-card">
    <h2>🎓 Bates Tech Student Advisor</h2>
    <p>Your AI-powered guide to programs, admissions, financial aid, and veteran benefits</p>
</div>
""", unsafe_allow_html=True)

# Initialize orchestrator (cached to avoid reloading)
@st.cache_resource
def get_orchestrator():
    """Initialize and cache the multi-agent orchestrator."""
    return MultiAgentOrchestrator()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.pending_query = None

if "orchestrator" not in st.session_state:
    with st.spinner("🚀 Initializing advisor agents..."):
        st.session_state.orchestrator = get_orchestrator()
        st.session_state.session_id = st.session_state.orchestrator.session_manager.create_session()

# Sidebar (must be before main chat logic to set pending_query)
with st.sidebar:
    st.markdown("### ❓ Frequently Asked Questions")
    
    faq_tab = st.radio("Category", ["📚 Programs", "📝 Admissions", "💰 Financial Aid", "🎖️ Veterans"], label_visibility="collapsed")
    
    if faq_tab == "📚 Programs":
        program_questions = [
            "What certificate and degree programs does Bates Technical College offer?",
            "What are the prerequisites for the Data Science program?",
            "Does the Welding program include hands-on lab experience?",
            "Are there online or hybrid options available?"
        ]
        for q in program_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.pending_query = q
    
    elif faq_tab == "📝 Admissions":
        admissions_questions = [
            "What are the minimum GPA and test score requirements?",
            "What documents do I need to submit with my application?",
            "What is the application deadline?",
            "Do you offer campus tours or information sessions?"
        ]
        for q in admissions_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.pending_query = q
    
    elif faq_tab == "💰 Financial Aid":
        financial_questions = [
            "What types of financial aid are available?",
            "How do I apply for FAFSA, and what is the priority filing deadline?"
        ]
        for q in financial_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.pending_query = q
    
    elif faq_tab == "🎖️ Veterans":
        veterans_questions = [
            "How do I use my GI Bill benefits at Bates?",
            "What financial aid is available for veterans?",
            "Are there any veteran-specific scholarships or grants?"
        ]
        for q in veterans_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.pending_query = q
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = st.session_state.orchestrator.session_manager.create_session()
        st.session_state.pending_query = None
        st.rerun()
    
    st.divider()
    st.markdown("### 🤖 About")
    st.markdown("""
    This advisor uses AI to answer questions about:
    - 📚 Programs & Courses
    - 📝 Admissions
    - 💰 Financial Aid
    
    Powered by RAG + Multi-Agent System
    """)

# Display chat history
for message in st.session_state.messages:
    escaped_content = html.escape(message["content"]).replace("\n", "<br>")
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-bubble user-bubble">
            {escaped_content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-bubble assistant-bubble">
            {escaped_content}
        </div>
        """, unsafe_allow_html=True)

# Get input from chat box OR pending query from sidebar
prompt = st.chat_input("Ask about programs, admissions, or financial aid...")

# Check if there's a pending query from sidebar button
if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

# Process the query
if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    escaped_prompt = html.escape(prompt).replace("\n", "<br>")
    st.markdown(f"""
    <div class="chat-bubble user-bubble">
        {escaped_prompt}
    </div>
    """, unsafe_allow_html=True)
    
    # Generate response
    with st.spinner("Thinking..."):
        response = st.session_state.orchestrator.process_query(
            query=prompt,
            session_id=st.session_state.session_id
        )
        
        # Display assistant response
        escaped_response = html.escape(response["response"]).replace("\n", "<br>")
        st.markdown(f"""
        <div class="chat-bubble assistant-bubble">
            {escaped_response}
        </div>
        """, unsafe_allow_html=True)
        
        # Show routing info in expander
        if response.get("agents_used"):
            with st.expander("ℹ️ Response Details"):
                st.write(f"**Agents used:** {', '.join(response['agents_used'])}")
                st.write(f"**Sources:** {response.get('sources_count', 'N/A')}")
                st.write(f"**Response time:** {response.get('response_time_ms', 'N/A')}ms")
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response["response"]})
    st.rerun()
