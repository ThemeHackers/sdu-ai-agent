
import os
import json
import glob
import time
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.ingest import extract_content, recursive_split_text
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY not found.")
    exit(1)

client = genai.Client(api_key=API_KEY)

def generate_qa_pair(context_text):
    prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านการสร้างข้อมูลทดสอบสำหรับระบบ AI (QA Generation Expert)
    
    Context (ข้อมูลอ้างอิง):
    "{context_text}"
    
    ภารกิจ:
    สร้างคำถาม-คำตอบ (Question-Answer Pair) จำนวน 1 ข้อ ที่วัดความเข้าใจจาก Context นี้
    โดยมีเงื่อนไขดังนี้:
    1. คำถามต้องเป็นภาษาไทยที่คนทั่วไปใช้ถาม (Natural Language) ไม่ใช่คำถามแบบข้อสอบ
    2. คำตอบต้องถูกต้องตาม Context ที่ให้เท่านั้น ห้ามเอาความรู้นอกเหนือจากนี้มาตอบ
    3. ถ้า Context นี้เป็นแค่หัวข้อหรือข้อมูลที่ไม่สมบูรณ์ ให้ตอบว่า "N/A" ทั้งคำถามและคำตอบ
    
    Output Format (JSON Only):
    {{
        "question": "คำถามภาษาไทย...",
        "answer": "คำตอบภาษาไทย..."
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        return json.loads(response.text)
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print("\n🚨 API Quota Exceeded (429). Stopping process immediately.")
            sys.exit(1)
        print(f"Error generating QA: {e}")
        return None

def main():
    data_dir = "data/processed"
    output_file = "data/synthetic_dataset.json"
    
    all_files = glob.glob(os.path.join(data_dir, "*.md"))
    print(f"Found {len(all_files)} files.")
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"Loaded {len(dataset)} existing items.")
        except:
             print("Could not load existing dataset, starting fresh.")
             dataset = []
    else:
        dataset = []
    
    existing_contexts = {item.get("context_sample", "")[:50] for item in dataset} # Simple dedup

    for file_path in all_files:
        print(f"Processing {file_path}...")
        results = extract_content(file_path)
        
        for item in results:
            text = item["text"]
            chunks = recursive_split_text(text)
            
            for i, chunk in enumerate(chunks):
                if len(chunk) < 100: 
                    continue
                
                # Check duplication
                if chunk[:50] in existing_contexts:
                     print(f"  Skipping chunk {i} (already exists)...")
                     continue

                # Reconstruct chunk ID to match ingest.py
                # Format: filename_page_sheet_i -> sanitized
                # For md files, page and sheet are empty strings.
                raw_id = f"{os.path.basename(file_path)}___{i}"
                import re
                chunk_id = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_id)
                
                print(f"  Generating QA for chunk {i} (ID: {chunk_id})...")
                qa = generate_qa_pair(chunk)
                
                if qa and qa.get("question") != "N/A" and qa.get("answer") != "N/A":
                    dataset.append({
                        "query": qa["question"],
                        "reference_answer": qa["answer"],
                        "relevant_ids": [chunk_id], 
                        "source_file": os.path.basename(file_path),
                        "context_sample": chunk[:200]
                    })
                    
                    # Save incrementally
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
                
                time.sleep(2) # Avoid rate limits (increased to 2s)

    print(f"✅ Finished. Total {len(dataset)} QA pairs. Saved to {output_file}")

if __name__ == "__main__":
    main()
