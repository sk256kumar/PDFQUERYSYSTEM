# PDF Query System  

## 📌 Overview  
The **PDF Query System** enables users to upload PDFs and query their content in natural language.  
It extracts text, processes queries using NLP/LLMs, and provides accurate answers or summaries.  

---

## 🚀 Features  
- PDF text extraction with `pdfplumber`  
- Natural language question answering  
- Summarization support  
- Simple interface  

---

## 🛠️ Tech Stack  
- **Language:** Python  
- **Libraries:** `pdfplumber`, `langchain`, `openai`/`gemini`, `streamlit`/`tkinter`  
- **Environment File:** `.env` for API keys  

---

## 📂 Project Files  
DF-Query-System/
│── app.py # Main application file
│── .env # Environment variables (API keys, config)
│── requirements.txt # Python dependencies
│── README.md # Documentation


---

## ⚙️ Installation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/pdf-query-system.git
   cd pdf-query-system
Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # For Linux/Mac  
venv\Scripts\activate      # For Windows  
Install dependencies:

pip install -r rr.txt

Add your API keys in .env:

OPENAI_API_KEY=your_api_key_here

run the Application

For Streamlit UI:

streamlit run app.py
GEMINI_API_KEY=your_api_key_here
