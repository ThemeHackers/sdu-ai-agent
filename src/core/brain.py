import os
import logging
import chromadb
import requests
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from dotenv import load_dotenv

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')
load_dotenv()

class GoogleGenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config={'task_type': 'RETRIEVAL_DOCUMENT'}
                )
                embeddings.append(response.embeddings[0].values)
            except Exception as e:
                embeddings.append([0.0]*768)
        return embeddings

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                if response.status_code == 200:
                    embeddings.append(response.json()["embedding"])
                else:
                    embeddings.append([0.0]*768)
            except Exception:
                embeddings.append([0.0]*768)
        return embeddings

class SmartBrain:
    def __init__(self, collection_name: str = "sdu_knowledge_v3"):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.db_path = "./data/chroma_db_v3"
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Provider Configuration
        if self.provider == "ollama":
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model_name = os.getenv("OLLAMA_MODEL", "llama3")
            self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            
            # Use a separate collection for Ollama to avoid embedding conflicts
            if collection_name == "sdu_knowledge_v3":
                collection_name = f"sdu_knowledge_ollama_{self.embedding_model.replace('-', '_')}"
                
            self.client = None # Not used for Ollama directly
            self.ef = OllamaEmbeddingFunction(base_url=self.ollama_base_url, model_name=self.embedding_model)
            print(f"🧠 SmartBrain initialized using OLLAMA ({self.model_name})")
            print(f"   Collection: {collection_name}")
            
        else: # Default to Gemini
            self.api_key = os.getenv("GEMINI_API_KEY")
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
                self.model_name = "gemini-2.5-flash"
                self.ef = GoogleGenAIEmbeddingFunction(api_key=self.api_key)
                print(f"🧠 SmartBrain initialized using GEMINI ({self.model_name})")
            else:
                self.client = None
                self.ef = None
                print("⚠️ Warning: GEMINI_API_KEY not found. SmartBrain is disabled.")

        try:
            if self.ef:
                self.collection = self.chroma_client.get_or_create_collection(
                    name=collection_name, 
                    embedding_function=self.ef
                )
            else:
                self.collection = None
        except Exception as e:
            print(f"❌ Error initializing collection: {e}")
            self.collection = None

    def _generate_content(self, system_instruction: str, contents: list, temperature: float = 0.3) -> str:
        """Internal helper to route generation requests."""
        if self.provider == "ollama":
            # Combine system instruction and contents for Ollama
            full_prompt = f"{system_instruction}\n\n"
            for item in contents:
                if isinstance(item, str):
                    full_prompt += item + "\n"
                elif isinstance(item, dict) and "content" in item:
                    full_prompt += item["content"] + "\n"
            
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {"temperature": temperature}
                    }
                )
                if response.status_code == 200:
                    return response.json().get("response", "")
                return ""
            except Exception as e:
                logging.error(f"Ollama generation error: {e}")
                return ""
        
        elif self.client: # Gemini
            try:
                # Filter contents to be just strings or valid parts for Gemini
                processed_contents = []
                for item in contents:
                     if isinstance(item, str):
                         processed_contents.append(item)
                     elif isinstance(item, dict) and "content" in item:
                         processed_contents.append(item["content"]) # Simplification
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[system_instruction] + processed_contents,
                    config={'temperature': temperature}
                )
                return response.text
            except Exception as e:
                logging.error(f"Gemini generation error: {e}")
                return ""
        return ""

    def expand_query(self, query: str) -> str:
        if not self.ef: return query # No provider
        
        system_prompt = """
        คุณคือผู้เชี่ยวชาญด้านการค้นหาข้อมูล (Search Expert)
        หน้าที่: แปลงคำถามสั้นๆ ของนักศึกษา ให้เป็น "คำค้นหา (Search Query)" ที่สมบูรณ์และครอบคลุมที่สุด
        
        ตัวอย่าง:
        - Input: "ลงทะเบียน"
        - Output: "ขั้นตอนการลงทะเบียนเรียน ช่วงเวลาการจองรายวิชา และเอกสารที่ต้องใช้ มหาวิทยาลัยสวนดุสิต"
        - Input: "ค่าเทอม"
        - Output: "อัตราค่าธรรมเนียมการศึกษา ค่าเทอมตลอดหลักสูตร สำหรับนักศึกษาปริญญาตรี"
        
        คำสั่ง: ตอบกลับเฉพาะ "คำค้นหาที่ขยายความแล้ว" เท่านั้น ห้ามมีคำอธิบายอื่น
        """
        response = self._generate_content(system_instruction=system_prompt, contents=[query], temperature=0.3)
        return response.strip() if response else query

    def retrieve(self, query: str, top_k: int = 15) -> list:
        if not self.collection:
            return []

        search_query = self.expand_query(query)

        try:
            results = self.collection.query(
                query_texts=[search_query],
                n_results=top_k
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []

            candidates = []
            for i in range(len(results['documents'][0])):
                candidates.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i] if 'distances' in results else 0
                })
            return candidates
        except Exception:
            return []

    def rerank(self, query: str, candidates: list, top_n: int = 5) -> list:
        if not candidates:
            return []
            
        fragments = ""
        for i, cand in enumerate(candidates):
            fragments += f"[{i}]: {cand['text'][:500]}\n---\n"

        system_rerank_prompt = f"""
        คุณคือ "RAG Ranker" หน้าที่ของคุณคือการอ่านรายการข้อมูลอ้างอิงและตัดสินว่าอันไหนเกี่ยวข้องกับ "คำถาม" มากที่สุด
        
        คำถาม: {query}
        
        รายการข้อมูล:
        {fragments}
        
        ภารกิจ:
        1. วิเคราะห์ความเกี่ยวข้องของแต่ละข้อมูลกับคำถาม
        2. เลือกผลลัพธ์ที่ตอบคำถามได้ตรงประเด็นที่สุดมาไม่เกิน {top_n} ลำดับ
        3. คืนค่าเป็น JSON Array ของ Index ที่เรียงลำดับจากเกี่ยวข้องมากที่สุดไปน้อยที่สุด
        
        Output Format:
        [index1, index2, index3]
        
        ตัวอย่าง:
        [5, 2, 0]
        """

        try:
            response = self._generate_content(system_instruction=system_rerank_prompt, contents=[], temperature=0.0)
            
            # Allow for some cleanup if the model outputs markdown code blocks
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:-3]
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:-3]
            
            import json
            indices = json.loads(clean_response)
            
            if not isinstance(indices, list):
                 logging.warning(f"Rerank output not a list: {response}")
                 return candidates[:top_n]

            reranked = []
            seen_indices = set()
            for idx in indices:
                try:
                    idx = int(idx)
                    if 0 <= idx < len(candidates) and idx not in seen_indices:
                        reranked.append(candidates[idx])
                        seen_indices.add(idx)
                except (ValueError, TypeError):
                    continue
            
            if not reranked:
                 return candidates[:top_n]

            return reranked
            
        except Exception as e:
            logging.error(f"Reranking error: {e}")
            return candidates[:top_n]

    def think(self, query: str, context: str, history: list = None):
        if not self.ef:
             yield "System Error: No LLM Provider configured."
             return

        if not context:
            context = "ยังไม่มีข้อมูลที่ชัดเจนในฐานความรู้ของมหาวิทยาลัยสวนดุสิตสำหรับคำถามนี้"

        system_instruction = """
        คุณคือ "พี่สวนดุสิต (SDU Smart Senior)" AI รุ่นพี่ที่ปรึกษาประจำมหาวิทยาลัยสวนดุสิต
        หน้าที่ของคุณคือการให้คำแนะนำน้องๆ นักศึกษาด้วยความถูกต้อง แม่นยำ และเป็นกันเอง

        Personality & Tone:
        - สุภาพ อ่อนโยน ขี้เล่นนิดๆ ให้รู้สึกเป็นกันเอง (ใช้สรรพนาม "พี่" กับ "น้อง")
        - ใช้ภาษาไทยที่สละสลวย อ่านง่าย ไม่เป็นทางการจนเกินไป (Semiprofessional)
        - แสดงความกระตือรือล้นที่จะช่วยเหลือ

        Strict Guidelines:
        1. **Context First:** ตอบคำถามโดยยึดข้อมูลจาก [Context ข้อมูลอ้างอิง] เป็นหลักเท่านั้น ห้ามมั่วข้อมูลขึ้นมาเองเด็ดขาด
        2. **Unknown Data:** ถ้าข้อมูลใน Context ไม่เพียงพอ ให้ตอบอย่างสุภาพว่า "ขอโทษด้วยนะครับ พี่อาจจะยังไม่มีข้อมูลส่วนนี้ในระบบ น้องอาจจะลองตรวจสอบที่หน้าเว็บคณะ/หน่วยงานโดยตรงอีกทีนะครับ"
        3. **Safety:** ห้ามตอบคำถามที่เกี่ยวกับ การเมือง, ความรุนแรง, เรื่องเพศ, หรือสิ่งผิดกฎหมาย
        4. **Structure:** จัดรูปแบบคำตอบให้อ่านง่าย (ใช้ Bullet points, ตัวหนา) ถ้าคำตอบยาว

        Goal: ทำให้น้องนักศึกษารู้สึกอุ่นใจและได้คำตอบที่ครบถ้วนที่สุด
        """
        
        user_message_content = f"[Context ข้อมูลอ้างอิง]:\n{context}\n\n[คำถาม]: {query}"
        messages = [{"role": "user", "content": user_message_content}]
        
        if history:
            full_history = []
            for h in history[-4:]: 
                full_history.append({"role": h["role"], "content": h["content"]})
            messages = full_history + messages

        if self.provider == "ollama":
            # Direct generation for Ollama
            full_text = self._generate_content(system_instruction, messages, temperature=0.3)
            # Simulate streaming
            chunk_size = 5
            for i in range(0, len(full_text), chunk_size):
                yield full_text[i:i+chunk_size]
        else:
            # Gemini with True Streaming
            try:
                processed_contents = [system_instruction] + [m["content"] for m in messages]
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=processed_contents,
                    config={
                        'temperature': 0.3,
                        'max_output_tokens': 800
                    },
                    stream=True
                )
                
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    yield "ขออภัยครับ พลังงานพี่สวนดุสิตหมดชั่วคราว (Rate Limit Exceeded) กรุณารอสักครู่แล้วถามใหม่นะครับ"
                else:
                    yield f"ขออภัยครับ พี่สวนดุสิตเกิดอาการมึนงงชั่วคราว ({error_msg})"

if __name__ == "__main__":
    brain = SmartBrain()
    q = "จุดเด่นของ ม.สวนดุสิต คืออะไร"
    # Testing only
    print("Testing generate...")
    try:
        candidates = brain.retrieve(q, top_k=10)
        reranked = brain.rerank(q, candidates, top_n=3)
        final_context = "\n\n".join([f"[ข้อมูลจาก: {c['metadata'].get('source', 'Unknown')}]\n{c['text']}" for c in reranked])
        print("Response:")
        for chunk in brain.think(q, final_context):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error in main: {e}")
