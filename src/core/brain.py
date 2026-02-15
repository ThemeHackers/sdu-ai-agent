import os
import logging
import chromadb
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
                print(f"Embedding failed: {e}")
                embeddings.append([0.0]*3072)
        return embeddings

class SmartBrain:
    def __init__(self, collection_name: str = "sdu_knowledge_v3"):
        self.db_path = "./data/chroma_db_v3"
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = "gemini-2.5-flash"
            self.ef = GoogleGenAIEmbeddingFunction(api_key=self.api_key)
        else:
            self.client = None
            self.ef = None

        try:
            if self.ef:
                self.collection = self.chroma_client.get_or_create_collection(
                    name=collection_name, 
                    embedding_function=self.ef
                )
            else:
                self.collection = None
        except Exception:
            self.collection = None

    def expand_query(self, query: str) -> str:
        """
        Uses Gemini to expand the user's short query into a more detailed search query.
        This improves retrieval accuracy significantly.
        """
        if not self.client: return query
        
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
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_prompt, query],
                config={'temperature': 0.3}
            )
            expanded = response.text.strip()
          
            return expanded
        except Exception:
            return query

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




        fragments = ""
        for i, cand in enumerate(candidates):
            fragments += f"[{i}]: {cand['text'][:500]}\n---\n"

        system_rerank_prompt = f"""
        คุณคือ "RAG Ranker" หน้าที่ของคุณคือการอ่านรายการข้อมูลอ้างอิงและตัดสินว่าอันไหนเกี่ยวข้องกับ "คำถาม" มากที่สุด
        
        คำถาม: {query}
        
        รายการข้อมูล:
        {fragments}
        
        ภารกิจ:
        - เลือกผลลัพธ์ที่ตอบคำถามได้ตรงประเด็นที่สุดมา {top_n} ลำดับ
        - คืนค่าออกมาเป็นลิตส์ของ "หมายเลขลำดับ" (ตัวเลขเท่านั้น) เรียงจากเกี่ยวข้องมากไปหาน้อย
        - เช่น: 5, 2, 0, 8, 1
        - ตอบเฉพาะตัวเลขคอมมา (,) เท่านั้น ห้ามบรรยายอื่น
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_rerank_prompt],
                config={'temperature': 0.0}
            )
            order_text = response.text.strip()

            indices = [int(idx.strip()) for idx in order_text.split(",") if idx.strip().isdigit()]
            
            reranked = []
            for idx in indices:
                if 0 <= idx < len(candidates):
                    reranked.append(candidates[idx])
            

            if len(reranked) < top_n:
                for cand in candidates:
                    if cand not in reranked:
                        reranked.append(cand)
                    if len(reranked) >= top_n: break
            
            return reranked[:top_n]
        except Exception as e:
            logging.error(f"Reranking error: {e}")
            return candidates[:top_n]

    def think(self, query: str, context: str, history: list = None):
        """
        Generates a streaming response from Gemini.
        Returns a generator object that yields chunks of text.
        """
        if not self.client:
            yield "System Error: API Key missing."
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
        
        messages = [
            {"role": "user", "content": f"[Context ข้อมูลอ้างอิง]:\n{context}\n\n[คำถาม]: {query}"}
        ]
        
        if history:
            full_history = []
            for h in history[-4:]: 
                full_history.append({"role": h["role"], "content": h["content"]})
            messages = full_history + messages

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_instruction] + [m["content"] for m in messages],
                config={
                    'temperature': 0.3,
                    'max_output_tokens': 800
                }
            )
            
            full_text = response.text
            chunk_size = 5
            for i in range(0, len(full_text), chunk_size):
                yield full_text[i:i+chunk_size]
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                yield "ขออภัยครับ พลังงานพี่สวนดุสิตหมดชั่วคราว (Rate Limit Exceeded) กรุณารอสักครู่แล้วถามใหม่นะครับ"
            else:
                yield f"ขออภัยครับ พี่สวนดุสิตเกิดอาการมึนงงชั่วคราว ({error_msg})"

if __name__ == "__main__":
    brain = SmartBrain()
    q = "จุดเด่นของ ม.สวนดุสิต คืออะไร"
    candidates = brain.retrieve(q, top_k=10)
    reranked = brain.rerank(q, candidates, top_n=3)
    final_context = "\n\n".join([f"[ข้อมูลจาก: {c['metadata'].get('source', 'Unknown')}]\n{c['text']}" for c in reranked])
    print(brain.think(q, final_context)['text'])
