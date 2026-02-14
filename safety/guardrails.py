import re
from html import escape

class SafetyGuardrails:
    def __init__(self):
        self.banned_keywords = [
            "hack", "exploit", "ddos", "jailbreak",
            "porn", "xxx", "gambling", 
            "hate speech", "racist"
        ]
        
    def check_jailbreak(self, prompt: str):
        prompt_lower = prompt.lower()
        for keyword in self.banned_keywords:
            if keyword in prompt_lower:
                return True, f"Found banned keyword: {keyword}"
        return False, "Safe"
        
    def validate_output(self, response: str):
        if not response:
            return False, "Empty response"
            
        response_lower = response.lower()
        if "i cannot fulfill" in response_lower and "safety" in response_lower:
            return False, "Model refusal"
            
        return True, "Safe"
        
    def sanitize_html(self, text: str):
        if not text: return ""
        return escape(text)
