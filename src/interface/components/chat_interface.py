import streamlit as st
import time
from safety.guardrails import SafetyGuardrails

def render_hero_section():
    st.markdown("""
    <div class="hero-box">
        <div class="hero-icon">✨</div>
        <div class="hero-content">
            <h1>Hello, Student</h1>
            <p style="color: #C4C7C5; font-size: 1.1rem; margin-top: 8px; max-width: 600px;">
                I'm your SDU AI Assistant. Ask me anything about curriculum, admissions, or campus life.
            </p>
            <div class="status-pill">
                <span style="display:inline-block; width:6px; height:6px; background:#81C995; border-radius:50%;"></span>
                System Online
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = [] 

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "✨"):
            st.markdown(msg["content"])

def handle_user_input(brain):
    if prompt := st.chat_input("Ask about Suan Dusit University..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="✨"):
            thinking_placeholder = st.empty()
            
            def update_thinking(step_text):
                thinking_placeholder.markdown(f"""
                <div class="thinking-box">
                    <div class="thinking-dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                    <div>
                        <div class="thinking-text">{step_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            guard = SafetyGuardrails()
            is_jailbreak, risk_msg = guard.check_jailbreak(prompt)
            
            if is_jailbreak:
                full_response = "I cannot fulfill this request due to safety guidelines."
                thinking_placeholder.markdown(f"🚨 {full_response}")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.stop()

            try:
                update_thinking("Searching University Knowledge Base...")
                candidates = brain.retrieve(prompt, top_k=10)
                
                update_thinking("Finding Most Relevant Info...")
                reranked_candidates = brain.rerank(prompt, candidates, top_n=4)
                
                context = "\n\n".join([
                    f"[Source: {c['metadata'].get('source', 'Unknown')}]\n{c['text']}" 
                    for c in reranked_candidates
                ])
                
                update_thinking("Generating Helpful Response...")
                
                full_response = ""
                response_generator = brain.think(prompt, context, history=st.session_state.messages[:-1])
                
                thinking_placeholder.empty()
                message_placeholder = st.empty()
                
                for chunk in response_generator:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.005) 
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"Error: {str(e)}"
                st.error(full_response)
            finally:
                thinking_placeholder.empty()

            is_safe, safety_msg = guard.validate_output(full_response)
            if not is_safe:
                blocked_msg = f"Response blocked by safety policy ({safety_msg})."
                message_placeholder.markdown(blocked_msg)
                full_response = blocked_msg

            
            
            if reranked_candidates and is_safe: 
                st.markdown("###### Sources")
                cols = st.columns(len(reranked_candidates) if len(reranked_candidates) < 3 else 2)
                for i, cand in enumerate(reranked_candidates[:4]):
                    col = cols[i % 2] if len(reranked_candidates) >= 2 else cols[0]
                    with col:
                        source_name = guard.sanitize_html(cand['metadata'].get('source', 'Unknown'))
                        clean_text = guard.sanitize_html(cand['text'])
                        preview = clean_text[:150] + "..." if len(clean_text) > 150 else clean_text
                        
                        st.markdown(f"""
                        <div class="citation-box">
                            <div class="citation-header">
                                <span>📄</span> {source_name}
                            </div>
                            <div class="citation-content">{preview}</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
