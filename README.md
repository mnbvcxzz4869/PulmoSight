# 🩻 PulmoSight

PulmoSight is a **web-based Clinical Decision Support System (CDSS)** designed to assist healthcare professionals in **tuberculosis (TB) screening** using **chest X-ray images and clinical text input**.  
The system integrates a **multimodal MedGemma model** fine-tuned for TB-related image analysis with an **interactive Streamlit interface** for image interpretation and clinical question–answering (QnA).

> ⚠️ **Professional Use Only**  
> This system is intended exclusively for use by qualified healthcare professionals.  
> It does **not** provide autonomous diagnoses or treatment recommendations.

---

## 🔍 System Overview

PulmoSight combines:
- Multimodal AI inference (image + text)
- Domain-specific clinical prompt engineering
- Interactive web-based user interface

The system is designed to:
- Prioritize **tuberculosis screening** in all interpretations
- Provide **probabilistic and cautious outputs**
- Emphasize the need for **clinical confirmation**

---

## 🧠 Model Description

### Base Model
- **MedGemma** (multimodal medical foundation model)
- Supports both **medical text** and **chest X-ray image** processing

### Fine-Tuning Strategy
- Fine-tuning is focused **only on the vision encoder**
- General clinical knowledge of the language model is preserved
- Implemented using **Low-Rank Adaptation (LoRA)**

---

## 🏋️ Training Details

- **Platform**: Google Colab  
- **GPU**: NVIDIA A100  
- **Dataset**: [Chest X-ray Tuberculosis Dataset (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

📓 **Training Notebook**  
https://colab.research.google.com/drive/1HGlNjgoMcmMG_XmztjDgN2msGy0rLUET?usp=sharing

🤗 **Fine-Tuned Model (Hugging Face)**  
https://huggingface.co/mnbvcxzz4869/medgemma-tb-it-lora-tb

---

## 🖥️ System Architecture

PulmoSight consists of three main components:

1. **Frontend (Streamlit UI)**  
   - Web-based interface for healthcare professionals  
   - Chest X-ray upload and visualization  
   - Clinical text-based QnA interaction  

2. **Inference Engine**  
   - MedGemma model with LoRA-adapted vision encoder  
   - Performs chest X-ray analysis and clinical reasoning  
   - Integrates multimodal context consistently  

3. **Prompt & Context Management**  
   - Separate prompts for image-based and text-based inputs  
   - Ensures TB screening is prioritized  
   - Outputs are probabilistic and clinically cautious  

---

## 🧩 Prompt Design

PulmoSight uses **different prompts** depending on input modality:

- **Image-Based Prompt**
  - Automatically triggered when a chest X-ray is uploaded
  - Focused on TB screening and radiological patterns

- **Text-Based Prompt**
  - Used during QnA interactions
  - Incorporates image analysis results as conversational context

This design ensures consistent, safe, and clinically aligned system behavior.

---

## 🚀 How to Run the Application

### 1. Clone the Repository
```bash
git clone https://github.com/mnbvcxzz4869/PulmoSight.git
cd PulmoSight
```

### 2. Install Depedencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
