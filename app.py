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
- Assess tuberculosis risk factors
- Suggest appropriate diagnostic steps (e.g., imaging, laboratory tests)

[CONSTRAINTS]
- Do NOT provide a definitive diagnosis
- Do NOT prescribe treatment
- Use cautious, non-absolute language
- Always emphasize the need for clinical correlation
"""

IMAGE_CDSS_PROMPT = """
[ROLE]
You are PulmoSight, a Clinical Decision Support System (CDSS) specialized in chest radiograph interpretation.
Your primary function is to screen for Pulmonary Tuberculosis (TB) signatures before evaluating other thoracic pathologies.

[PRIMARY CLINICAL PRIORITY]
Pulmonary Tuberculosis (TB) screening MUST be explicitly evaluated first.

[CLINICAL CONTEXT]
{context}

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
- **Other Possible Etiologies:** (If not TB, suggest what other pathology the patterns resemble).
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

# Context Builder
def build_model_context(max_turns=6):
    msgs = st.session_state.messages[-max_turns * 2:]
    lines = []
    for m in msgs:
        role = "Clinician" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    if st.session_state.image_report:
        lines.append("Assistant (Image Report):")
        lines.append(st.session_state.image_report)
    return "\n".join(lines) if lines else "No prior clinical information."

# Image Analysis
def analyze_image(image):
    context = build_model_context()
    prompt = IMAGE_CDSS_PROMPT.format(context=context)

    convo = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Analyze this chest X-ray."}]
    }]

    chat = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True, system_prompt=prompt
    )

    inputs = processor(text=chat, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1200)

    return processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# Side Bar
with st.sidebar:
    st.title("📁 Upload Chest X-ray")
    uploaded = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])
    if uploaded:
        file_id = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.last_file_id != file_id:
            st.session_state.current_image = Image.open(uploaded).convert("RGB")
            st.session_state.image_report = None
            st.session_state.last_file_id = file_id

        st.image(st.session_state.current_image, use_container_width=True)
        st.caption(uploaded.name)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.image_report = None
        st.session_state.current_image = None
        st.session_state.last_file_id = None

# Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
user_input = st.chat_input("Type clinical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # If image exists and has not been analyzed yet → run IMAGE_CDSS_PROMPT first
    if st.session_state.current_image and st.session_state.image_report is None:
        with st.chat_message("assistant"):
            with st.spinner("🩻 Analyzing chest X-ray..."):
                try:
                    report = analyze_image(st.session_state.current_image)
                    st.session_state.image_report = report
                    st.markdown(report)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"## 🩻 Chest X-ray Analysis\n\n{report}"
                    })
                finally:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

    # Then process user input with TEXT_CDSS_PROMPT
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            context = build_model_context()
            text_prompt = TEXT_CDSS_PROMPT.format(context=context)

            convo = [{"role": "system", "content": text_prompt}]
            convo.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

            chat = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=chat, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=1200)

            response = processor.decode(
                output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
