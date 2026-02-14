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

    def retrieve(self, query: str, top_k: int = 15) -> list:
        if not self.collection:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
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

    def think(self, query: str, context: str, history: list = None) -> str:
        if not self.client:
            return "System Error: API Key missing."

        if not context:
            context = "ยังไม่มีข้อมูลที่ชัดเจนในฐานความรู้ของมหาวิทยาลัยสวนดุสิตสำหรับคำถามนี้"

        system_instruction = """
        คุณคือ "พี่สวนดุสิต (SDU Smart Senior)" AI ผู้ช่วยอัจฉริยะที่รอบรู้และเป็นกันเองที่สุดในมหาวิทยาลัยสวนดุสิต
        
        บุคลิกภาพ:
        - สุภาพ ใจดี ทันสมัย และเชื่อถือได้
        - แทนตัวเองว่า "พี่" หรือ "พี่สวนดุสิต"
        - ใช้หางเสียง "ครับ/ค่ะ" ตามความเหมาะสม (เน้นครับเป็นหลักสำหรับระบบกลาง)
        
        เป้าหมาย:
        - ช่วยให้คำแนะนำที่ถูกต้อง แม่นยำ และรวดเร็ว เกี่ยวกับข้อมูลมหาวิทยาลัย
        
        กฎเหล็ก (Strict Rules):
        1. **จำกัดความยาว (สำคัญมาก):** ตอบกลับให้กระชับที่สุด **ต้องมีความยาวระหว่าง 100 - 250 ตัวอักษรเท่านั้น** (รวมทั้งภาษาไทยและอังกฤษ)
        2. **ยึดตาม Context:** ตอบคำถามโดยอ้างอิงจาก [Context ข้อมูลอ้างอิง] ที่ให้มาเท่านั้น
        3. **ห้ามเดาข้อมูลสำคัญ:** หากใน Context ไม่มีข้อมูลที่น้องถาม ให้ตอบอย่างสุภาพว่า "ขออภัยครับ พี่ยังไม่มีข้อมูลส่วนนี้ในระบบครับ"
        4. **สรุปใจความ:** ตอบให้ตรงประเด็นทันที ไม่ต้องเกริ่นนำยืดเยื้อ
        5. **ลงท้ายด้วยความประทับใจ:** สั้นๆ เช่น "สู้ๆ นะครับ" หรือ "ยินดีเสมอครับ"
        """
        
        messages = [
            {"role": "user", "content": f"[Context ข้อมูลอ้างอิง]:\n{context}\n\n[คำถาม]: {query}"}
        ]
        
        if history:
            full_history = []
            current_tokens = 0
            # Simple token estimation: 1 token approx 4 chars
            # Reserve 2000 chars for system prompt + current context
            max_history_chars = 12000 
            
            for h in reversed(history):
                content_len = len(h["content"])
                if current_tokens + content_len > max_history_chars:
                    break
                full_history.insert(0, {"role": h["role"], "content": h["content"]})
                current_tokens += content_len
                
            messages = full_history + messages

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_instruction] + [m["content"] for m in messages],
                config={
                    'temperature': 0.2,
                    'max_output_tokens': 200
                }
            )
            
            usage = {}
            if response.usage_metadata:
                usage = {
                    "prompt_token_count": response.usage_metadata.prompt_token_count,
                    "candidates_token_count": response.usage_metadata.candidates_token_count,
                    "total_token_count": response.usage_metadata.total_token_count
                }

            return {
                "text": response.text,
                "usage": usage
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                friendly_msg = "ขออภัยครับ พลังงานพี่สวนดุสิตหมดชั่วคราว (Rate Limit Exceeded) กรุณารอสักครู่แล้วถามใหม่นะครับ"
            else:
                friendly_msg = f"ขออภัยครับ พี่สวนดุสิตเกิดอาการมึนงงชั่วคราว ({error_msg})"
            
            return {
                "text": friendly_msg,
                "usage": {}
            }

if __name__ == "__main__":
    brain = SmartBrain()
    q = "จุดเด่นของ ม.สวนดุสิต คืออะไร"
    candidates = brain.retrieve(q, top_k=10)
    reranked = brain.rerank(q, candidates, top_n=3)
    final_context = "\n\n".join([f"[ข้อมูลจาก: {c['metadata'].get('source', 'Unknown')}]\n{c['text']}" for c in reranked])
    print(brain.think(q, final_context)['text'])
