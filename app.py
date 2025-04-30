import os
import streamlit as st
import pdfplumber
from docx import Document
from weasyprint import HTML
import tempfile
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set environment variables for caching
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Function to download model weights from GitHub release
@st.cache_resource
def download_model_weights():
    model_dir = "/tmp/model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.safetensors")
    
    if not os.path.exists(model_path):
        url = "https://github.com/woyeso/assignment-grader-app/releases/download/v1.0.0/model.safetensors"
        logger.info(f"Downloading model weights from {url}")
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return model_dir

# Load model and tokenizer
@st.cache_resource
def load_model():
    base_model_name = "unsloth/Llama-3.2-3B-Instruct"
    adapter_model_name = "woyeso/fine_tuned_llama_3_2_assignment_grader"
    hf_token = os.getenv("HF_TOKEN")
    
    # Download base model weights
    model_dir = download_model_weights()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_model_name,
        token=hf_token if hf_token else None
    )
    
    # Load base model from local weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load PEFT adapters
    model = PeftModel.from_pretrained(base_model, adapter_model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to evaluate submission
def evaluate_submission(submission_text, rubric, project_type, subcomponent):
    prompt = (
        f"Can you evaluate my project submission for Subcomponent {subcomponent} "
        f"in a {project_type} project.\nHere is the rubric: {rubric}\n"
        f"My submission is {submission_text}\nProvide a score out of 10 and detailed feedback."
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return feedback

# Streamlit app
st.title("Assignment Grader App")

# File upload
uploaded_file = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"])
project_type = st.selectbox("Project Type", ["Group (P1)", "Individual (P2)"])
school_name = st.text_input("School Name (Optional)")
group_number = st.text_input("Group Number (Optional)")

# Manual text input as fallback
manual_text = st.text_area("Or enter your submission text manually (optional)")

if st.button("Evaluate"):
    if uploaded_file or manual_text:
        # Extract text from file
        submission_text = ""
        if uploaded_file:
            if uploaded_file.name.endswith(".pdf"):
                with pdfplumber.open(uploaded_file) as pdf:
                    submission_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            elif uploaded_file.name.endswith(".docx"):
                doc = Document(uploaded_file)
                submission_text = "\n".join(para.text for para in doc.paragraphs)
        else:
            submission_text = manual_text

        if not submission_text:
            st.error("No text extracted from the file or provided manually.")
        else:
            # Evaluate submission
            rubric = "Score based on clarity, relevance, and completeness."
            subcomponent = "1.1"
            project_type_short = "P1" if project_type == "Group (P1)" else "P2"
            
            with st.spinner("Evaluating submission..."):
                feedback = evaluate_submission(submission_text, rubric, project_type_short, subcomponent)
            
            # Display results
            st.subheader("Evaluation Results")
            st.write(feedback)

            # PDF generation (temporarily disabled due to Fontconfig issues)
            pdf_available = False
            pdf_path = None
    else:
        st.error("Please upload a file or enter text manually.")