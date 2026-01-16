"""
Custom CSS Styles for CyberBron
Provides dark cybersecurity-themed styling.
"""


def get_custom_css():
    """
    Get custom CSS for CyberBron dark cybersecurity theme.
    
    Returns:
        CSS string to inject into Streamlit
    """
    return """
    <style>
    /* Main theme colors */
    :root {
        --cyber-green: #00ff88;
        --cyber-cyan: #00d4ff;
        --cyber-dark: #0d1117;
        --cyber-dark-light: #161b22;
        --cyber-text: #c9d1d9;
        --cyber-text-dim: #8b949e;
    }
    
    /* Dark background */
    .stApp {
        background-color: var(--cyber-dark);
        color: var(--cyber-text);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--cyber-dark-light);
        border-right: 1px solid var(--cyber-green);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: var(--cyber-text);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--cyber-dark-light);
        padding: 10px;
        border-radius: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--cyber-dark);
        color: var(--cyber-text);
        border: 1px solid var(--cyber-text-dim);
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--cyber-green);
        color: var(--cyber-dark);
        border-color: var(--cyber-green);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: var(--cyber-dark-light);
        border: 1px solid var(--cyber-text-dim);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stChatMessage[data-testid="user-message"] {
        border-left: 3px solid var(--cyber-cyan);
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        border-left: 3px solid var(--cyber-green);
    }
    
    /* Button styling */
    .stButton button {
        background-color: var(--cyber-dark-light);
        color: var(--cyber-green);
        border: 1px solid var(--cyber-green);
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: var(--cyber-green);
        color: var(--cyber-dark);
        box-shadow: 0 0 10px var(--cyber-green);
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: var(--cyber-dark-light);
        color: var(--cyber-text);
        border: 1px solid var(--cyber-text-dim);
        border-radius: 5px;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--cyber-green);
        box-shadow: 0 0 5px var(--cyber-green);
    }
    
    /* Cards/containers */
    .element-container div[data-testid="stVerticalBlock"] > div {
        background-color: var(--cyber-dark-light);
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Success/Info/Warning/Error */
    .stSuccess {
        background-color: rgba(0, 255, 136, 0.1);
        border-left: 3px solid var(--cyber-green);
    }
    
    .stInfo {
        background-color: rgba(0, 212, 255, 0.1);
        border-left: 3px solid var(--cyber-cyan);
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 3px solid #ffc107;
    }
    
    .stError {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 3px solid #ff0000;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--cyber-green);
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--cyber-green);
        font-size: 2em;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--cyber-dark-light);
        border: 1px solid var(--cyber-text-dim);
        border-radius: 5px;
        color: var(--cyber-text);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background-color: var(--cyber-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: var(--cyber-green);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: var(--cyber-cyan);
    }
    
    /* Links */
    a {
        color: var(--cyber-cyan);
        text-decoration: none;
    }
    
    a:hover {
        color: var(--cyber-green);
        text-shadow: 0 0 5px var(--cyber-green);
    }
    
    /* Code blocks */
    code {
        background-color: var(--cyber-dark-light);
        color: var(--cyber-green);
        padding: 2px 6px;
        border-radius: 3px;
        border: 1px solid var(--cyber-text-dim);
    }
    
    pre {
        background-color: var(--cyber-dark-light);
        border: 1px solid var(--cyber-text-dim);
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """


def apply_custom_css():
    """Apply custom CSS to Streamlit app."""
    import streamlit as st
    st.markdown(get_custom_css(), unsafe_allow_html=True)
