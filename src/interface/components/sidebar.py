import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <h2>SDU Agent</h2>
            <p>Smart Student Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 System Stats")
        
        st.markdown("""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-num">5k+</div>
                <div class="stat-label">Docs</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">24h</div>
                <div class="stat-label">Uptime</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.caption("Powered by Gemini 2.5 Pro")
        st.caption("Suan Dusit University")
