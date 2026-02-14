<div align="center">

# 🎓 SDU AI Agent — พี่สวนดุสิต

### ระบบ AI ผู้ช่วยอัจฉริยะสำหรับนักศึกษามหาวิทยาลัยสวนดุสิต

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/gallery)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Pro-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br>

> **พี่สวนดุสิต** คือผู้ช่วย AI ที่ใช้เทคโนโลยี **Retrieval-Augmented Generation (RAG)** ตอบคำถามเกี่ยวกับชีวิตในมหาวิทยาลัย การรับสมัคร และหลักสูตรได้อย่างรวดเร็วและแม่นยำ ขับเคลื่อนด้วย **Google Gemini 2.5 Pro**

</div>

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| 🧠 **Gemini 2.5 Pro** | ใช้โมเดล AI ล่าสุดจาก Google เป็นสมองหลักในการประมวลผลคำถาม |
| 📚 **RAG + ChromaDB** | ระบบค้นหาความรู้จากฐานข้อมูลเวกเตอร์ เพื่อตอบคำถามจากข้อมูลจริงของมหาวิทยาลัย |
| 🎯 **Smart Reranking** | จัดลำดับผลลัพธ์ด้วย prompt เฉพาะทาง เพื่อความเกี่ยวข้องสูงสุด |
| 🛡️ **Guardrails** | กรองเนื้อหาที่ไม่เหมาะสมและป้องกัน Jailbreak attempts |
| 🌙 **Premium UI** | ธีม Dark Mode สไตล์ Gemini พร้อม animation "Thinking..." และแสดงแหล่งอ้างอิง |

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology |
| :---: | :---: |
| **LLM** | Google Gemini 2.5 Pro |
| **Vector DB** | ChromaDB |
| **Framework** | Streamlit |
| **Language** | Python 3.10+ |
| **Document Parsing** | PyPDF · python-docx · openpyxl |

</div>

---

## ⚡ Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ThemeHackers/sdu-ai-agent.git
cd sdu-ai-agent
```

### 2️⃣ Set Up Environment Variables

สร้างไฟล์ `.env` ที่ root directory แล้วใส่ API Key ของ Google Gemini:

```env
GEMINI_API_KEY=your_api_key_here
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Ingest Knowledge Base

นำเข้าเอกสารและสร้าง Vector Database:

```bash
python3 src/core/ingest.py
```

### 5️⃣ Run the Application

```bash
python3 -m streamlit run src/interface/app.py
```

เปิดเบราว์เซอร์ไปที่ **`http://localhost:8501`** แล้วเริ่มสนทนากับ **พี่สวนดุสิต** ได้เลย! 🎉

---

## 📂 Project Structure

```
sdu-ai-agent/
├── 📁 src/
│   ├── 📁 core/
│   │   ├── brain.py          # 🧠 AI Logic & RAG Pipeline
│   │   └── ingest.py         # 📥 Document Ingestion & Vectorization
│   └── 📁 interface/
│       ├── app.py             # 🚀 Streamlit Entry Point
│       ├── assets/            # 🎨 Static Assets
│       └── components/        # 🧩 UI Components & Styling
├── 📁 data/
│   ├── raw/                   # 📄 Raw Documents
│   ├── processed/             # ✅ Processed Documents
│   └── chroma_db_v3/          # 💾 Vector Database
├── 📁 safety/
│   └── guardrails.py          # 🛡️ Content Filtering & Security
├── 📁 evaluation/
│   └── metrics.py             # 📊 Quality Metrics
├── .env                       # 🔑 Environment Variables
├── requirements.txt           # 📦 Python Dependencies
└── README.md                  # 📖 You are here!
```

---

## 🤝 Contributing

ยินดีต้อนรับทุก Contribution! สามารถ:

1. 🐛 เปิด **Issue** เพื่อรายงานบัก
2. 💡 เสนอ **Feature Request** สำหรับฟีเจอร์ใหม่
3. 🔧 ส่ง **Pull Request** เพื่อปรับปรุงโค้ด

---

## 📄 License

โปรเจกต์นี้อยู่ภายใต้ [MIT License](LICENSE)

---

<div align="center">

**Made with ❤️ by [ThemeHackers](https://github.com/ThemeHackers)**

</div>
