def get_custom_css():
    return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Prompt:wght@300;400;500;600&display=swap');

    :root {
        --bg-color: #131314;
        --sidebar-bg: #1E1F20;
        --card-bg: #2D2E31;
        --primary-gradient: linear-gradient(135deg, #4285F4, #9B72CB);
        --accent-color: #AECBFA;
        --text-high: #E3E3E3;
        --text-medium: #C4C7C5;
        --border-color: rgba(255, 255, 255, 0.1);
        --input-bg: #282A2C;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-color);
        font-family: 'Outfit', 'Prompt', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, button, input, textarea, label {
        font-family: 'Outfit', 'Prompt', sans-serif;
        color: var(--text-high);
    }
    
    [data-testid="stSidebarNav"] span, 
    [data-testid="collapsedControl"] span,
    i[class^="icon"],
    span[class^="icon"] {
         font-family: unset !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
    }

    .sidebar-brand {
        text-align: center;
        padding: 24px 12px;
        margin-bottom: 20px;
        border-bottom: 1px solid var(--border-color);
    }

    .sidebar-brand h2 {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.6rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .sidebar-brand p {
        color: var(--text-medium);
        font-size: 0.85rem;
        margin-top: 4px;
    }

    .stButton button {
        background: transparent;
        border: 1px solid var(--border-color);
        color: var(--text-high);
        border-radius: 8px;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        background: rgba(255,255,255,0.05);
        border-color: #4285F4;
        color: #4285F4;
    }

    .hero-box {
        text-align: left;
        padding: 40px;
        background: linear-gradient(180deg, rgba(66, 133, 244, 0.1) 0%, transparent 100%);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        margin-bottom: 40px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 16px;
    }

    .hero-icon {
        font-size: 48px;
        background: var(--card-bg);
        width: 72px;
        height: 72px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 8px;
    }

    .hero-content h1 {
        font-size: 2.4rem;
        font-weight: 600;
        background: linear-gradient(90deg, #FFFFFF, #AECBFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 100px;
        color: #81C995;
        font-size: 0.8rem;
        font-weight: 500;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 20px 0; 
    }

    [data-testid="stChatMessage"][data-testid="user"] {
        background: transparent;
    }
    
    [data-testid="stChatMessage"][data-testid="user"] .stMarkdown {
        background: #2D2E31;
        color: var(--text-high);
        padding: 12px 20px;
        border-radius: 20px;
        max-width: 80%;
        margin-left: auto;
        border: 1px solid var(--border-color);
    }

    [data-testid="stChatMessage"][data-testid="assistant"] .stMarkdown {
        background: transparent;
        color: var(--text-high);
        padding: 0 10px;
    }

    [data-testid="stChatInput"] {
        padding-bottom: 30px;
        background: transparent !important;
    }

    [data-testid="stChatInput"] > div {
        background-color: transparent !important;
        border: 1px solid var(--border-color);
        border-radius: 100px;
        color: var(--text-high);
        box-shadow: none !important;
        padding: 6px 16px;
        transition: border-color 0.2s ease;
    }
    
    [data-testid="stChatInput"] > div div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #4285F4;
        background-color: rgba(255,255,255,0.03) !important; 
    }

    [data-testid="stChatInput"] textarea {
        color: var(--text-high) !important;
        font-size: 1rem;
        background: transparent !important;
    }
    
    [data-testid="stChatInput"] span {
        color: var(--text-medium) !important;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 20px;
    }

    .stat-card {
        background: var(--card-bg);
        padding: 16px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    
    .stat-card:hover {
        border-color: #5F6368;
        transform: translateY(-2px);
    }

    .stat-num {
        font-size: 1.4rem;
        font-weight: 600;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: var(--text-medium);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    .citation-box {
        background: #1E1F20;
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
    }
    
    .citation-box:hover {
        border-color: #4285F4;
        background: #232527;
    }
    
    .citation-header {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #8AB4F8;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .citation-content {
        color: var(--text-medium);
        font-size: 0.85rem;
        line-height: 1.5;
    }

    @keyframes gradient-x {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .thinking-box {
        padding: 16px 24px;
        background: linear-gradient(-45deg, #1E1F20, #232527, #1E1F20, #282A2C);
        background-size: 400% 400%;
        animation: gradient-x 3s ease infinite;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        color: #E3E3E3;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .thinking-dots {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        animation: ai-pulse 1.4s infinite ease-in-out both;
    }
    
    .dot:nth-child(1) {
        background-color: #4285F4;
        animation-delay: -0.32s;
    }
    
    .dot:nth-child(2) {
        background-color: #9B72CB;
        animation-delay: -0.16s;
    }
    
    .dot:nth-child(3) {
        background-color: #34A853;
        animation-delay: 0;
    }
    
    @keyframes ai-pulse {
        0%, 80%, 100% { 
            transform: scale(0.6); 
            opacity: 0.6;
        }
        40% { 
            transform: scale(1.2); 
            opacity: 1; 
            box-shadow: 0 0 8px rgba(255,255,255,0.2);
        }
    }
    
    .thinking-text {
        font-weight: 500;
        background: linear-gradient(90deg, #E3E3E3, #AECBFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    
    .thinking-steps {
        font-size: 0.8rem;
        color: #9AA0A6;
        margin-left: auto;
</style>
"""
