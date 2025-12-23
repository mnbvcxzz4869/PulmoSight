import streamlit as st
import torch
import gc
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

torch.classes.__path__ = []

# Model Config
BASE_MODEL_ID = "google/medgemma-4b-it"
ADAPTER_MODEL_ID = "mnbvcxzz4869/medgemma-tb-it-lora-tb" 

# Streamlit Config
st.set_page_config(page_title="PulmoSight - TB Screening CDSS", layout="wide")

st.title("🩻 PulmoSight - TB Screening CDSS")
st.markdown("""
> ⚠️ **Professional Use Only**: This software is intended exclusively for use by qualified healthcare professionals.

*This system is a **Clinical Decision Support System (CDSS)** designed to support **clinical screening and assessment**. 
It does **not** provide autonomous diagnoses or treatment recommendations.*
""")

# Prompts
TEXT_CDSS_PROMPT = """
[ROLE]
You are PulmoSight, a Clinical Decision Support System (CDSS) specialized in chest radiograph interpretation.
Your primary function is to screen for Pulmonary Tuberculosis (TB) signatures before evaluating other thoracic pathologies.
You must only attribute elevated TB likelihood when radiographic or clinical findings specifically support it and clearly state when TB appears unlikely.

[CLINICAL CONTEXT]
{context}

[TASK]
Support healthcare professionals through clinical reasoning and discussion based on:
- Patient-provided clinical information
- Conversation context
- Any available imaging analysis

You may:
- Answer clinical questions
- Ask clarifying or follow-up questions
- Suggest appropriate diagnostic steps (e.g., imaging, laboratory tests)

[CONSTRAINTS]
- Do NOT provide a definitive diagnosis
- Do NOT prescribe treatment
- Use cautious, non-absolute language
- Always emphasize the need for clinical correlation
- When evidence does not support TB, explicitly state that TB is unlikely and prioritise alternative explanations consistent with the findings
"""

IMAGE_CDSS_PROMPT = """
[ROLE]
You are PulmoSight, a Clinical Decision Support System (CDSS) specialized in chest radiograph interpretation.
Your primary function is to screen for Pulmonary Tuberculosis (TB) signatures before evaluating other thoracic pathologies.
You must avoid over-attributing findings to TB; only report TB likelihood when supported by observed radiographic patterns.

[PRIMARY CLINICAL PRIORITY]
Pulmonary Tuberculosis (TB) screening MUST be explicitly evaluated first, confirming whether findings are supportive, equivocal, or unsupportive for TB before addressing other differentials.

[TASK DEFINITION]
Using the provided chest X-ray image:
1. Systematically assess radiographic features associated with pulmonary TB, explicitly stating their presence or absence.
2. Describe all observable radiographic findings using standard radiological terminology.
3. Synthesize these findings into a cautious clinical interpretation.
4. If TB-related findings are absent or equivocal, consider other possible thoracic conditions without deprioritizing TB screening.

[RESPONSE REQUIREMENTS]
- Use cautious, probabilistic language.
- Explicitly state uncertainty or image limitations when present.
- Clearly separate radiographic observations from clinical interpretation.
- When TB manifestations are absent, highlight this clearly and focus the interpretation on alternative pathologies that better explain the findings.

[SAFETY & LIMITATIONS]
- Do NOT provide a definitive diagnosis.
- Do NOT recommend specific treatments.
- Always require clinical correlation and further diagnostic confirmation.

[OUTPUT FORMAT]
Provide a structured radiological report with the following sections:
- **TB Screening Assessment:** (State clearly if findings are Suggestive, Equivocal, or Unlikely for TB, citing specific visual evidence).
- **Lung Fields:** (Describe opacities, texture, and location).
- **Mediastinum and Cardiac Silhouette:** (Assess size and contours).
- **Pleural Spaces:** (Check for effusions or thickening).
- **Other Possible Etiologies:** (Suggest what other pathology the patterns resemble).
- **Summary and Recommendations:** (Summarize key findings, emphasize TB screening results, and recommend next steps for clinical correlation and further testing).
"""

# Load Model
@st.cache_resource
def load_model():
    if torch.cuda.is_available():
        device = "cuda"
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        device = "cpu"
        quant = None

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quant,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL_ID)
    model.eval()
    return model, processor, device

with st.spinner("⏳ Loading model..."):
    model, processor, device = load_model()

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "image_report" not in st.session_state:
    st.session_state.image_report = None

if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

# Image Analysis
def analyze_image(image):
    convo = [
        {"role": "system", "content": IMAGE_CDSS_PROMPT},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Analyze this chest X-ray."}]
        }
    ]

    chat = processor.apply_chat_template(
        convo,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(text=chat, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1200)

    return processor.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

# Side Bar
with st.sidebar:
    st.title("📁 Chest X-ray Screening")

    uploaded = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

    if uploaded:
        file_id = f"{uploaded.name}-{uploaded.size}"

        if st.session_state.last_file_id != file_id:
            st.session_state.current_image = Image.open(uploaded).convert("RGB")
            with st.spinner("🩻 Analyzing X-ray..."):
                st.session_state.image_report = analyze_image(st.session_state.current_image)
            st.session_state.last_file_id = file_id

        st.image(st.session_state.current_image, use_container_width=True)
        st.caption(uploaded.name)

        st.markdown("### 🩻 Radiology Report")
        st.markdown(st.session_state.image_report)

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []

# Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
user_input = st.chat_input("Ask a clinical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            context_parts = []

            if st.session_state.image_report:
                context_parts.append("Radiology Report:")
                context_parts.append(st.session_state.image_report)

            for m in st.session_state.messages[-6:]:
                role = "Clinician" if m["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {m['content']}")

            context = "\n".join(context_parts) if context_parts else "No prior clinical data."

            prompt = TEXT_CDSS_PROMPT.format(context=context)

            convo = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "text", "text": user_input}]}
            ]

            chat = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=chat, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=1200)

            response = processor.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()