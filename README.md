# Patient---Symptom---matcher
📌 Overview

AI Symptom Checker is an intelligent healthcare web application that predicts possible diseases based on user-entered symptoms. It uses Natural Language Processing (NLP), sentence embeddings, and Large Language Model (LLM) integration to provide accurate predictions and simple medical explanations.

**🚀 Features**

🔎 Symptom-based disease prediction
🤖 LLM-powered medical explanations
📊 Similarity score visualization
🔐 Login system for user access
🎨 Modern and responsive UI (Streamlit)
⚡ Real-time predictions

**
🛠️ Technologies Used**
Python
Streamlit
Sentence Transformers (all-MiniLM-L6-v2)
Scikit-learn
NumPy, Pandas
Matplotlib
Groq API (LLM integration)


**🧠 How It Works**
User enters symptoms (e.g., fever, headache)
Input is converted into embeddings using NLP model
Cosine similarity is calculated with dataset
Most similar disease is predicted
LLM generates explanation in simple terms


**📂 Project Structure**
├── app.py
├── metadata.pkl
├── requirements.txt
├── templates/
├── static/

**
⚙️ Installation & Setup**


# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

🔑 Environment Variables

Set your Groq API key:

export GROQ_API_KEY=your_api_key

📊 Output
Predicted Disease
Similarity Score (%)
AI-based Medical Explanation

Screenshots:
<img width="1922" height="957" alt="image" src="https://github.com/user-attachments/assets/f845a0b3-2694-4084-9653-04ee7a73e796" />
<img width="1922" height="939" alt="image" src="https://github.com/user-attachments/assets/8cc259e9-54e5-43cc-95f0-ce1c2232db23" />
<img width="1922" height="955" alt="image" src="https://github.com/user-attachments/assets/9ac763b0-d3bb-4a3f-ab8f-f56e67687aee" />
<img width="1922" height="926" alt="image" src="https://github.com/user-attachments/assets/98228dbc-175d-4486-b779-85a229f7162d" />
<img width="1922" height="974" alt="image" src="https://github.com/user-attachments/assets/c22d3908-aa52-4195-b278-a0d386245b54" />
<img width="1922" height="949" alt="image" src="https://github.com/user-attachments/assets/b1c669d0-9ce4-447e-822b-996cdff59ad5" />





⚠️ Disclaimer

This application is for educational purposes only and should not be used as a substitute for professional medical advice.

👩‍💻 Author

Siddam Mounika

Python Developer | AI & Deep Learning Enthusiast
