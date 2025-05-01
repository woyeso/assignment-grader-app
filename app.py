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

# Set environment variables for caching and Fontconfig
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["FONTCONFIG_PATH"] = "/tmp/fontconfig"
os.environ["XDG_CACHE_HOME"] = "/tmp/xdg_cache"
os.makedirs("/tmp/huggingface_cache", exist_ok=True)
os.makedirs("/tmp/fontconfig", exist_ok=True)
os.makedirs("/tmp/xdg_cache", exist_ok=True)

# Inform users about initialization
st.info("Initializing the app... This may take a few minutes as the model weights are downloaded.")

# Function to download and reassemble model weights
@st.cache_resource
def download_model_weights():
    model_dir = "/tmp/model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Use unsloth/Llama-3.2-1B-Instruct to reduce memory usage (comment out to revert to 3B model)
    release_url = "https://github.com/woyeso/assignment-grader-app/releases/download/v1.1.0"
    files_to_download = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ]
    safetensors_shards = [
        ("model-00001-of-00002.safetensors", ["partaa", "partab"]),  # Adjust based on actual shards for 1B model
        ("model-00002-of-00002.safetensors", ["partaa", "partab"]),
    ]

    # For unsloth/Llama-3.2-3B-Instruct (uncomment if using 3B model with upgraded hardware)
    # release_url = "https://github.com/woyeso/assignment-grader-app/releases/download/v1.0.0"
    # safetensors_shards = [
    #     ("model-00001-of-00003.safetensors", ["partaa", "partab", "partac", "partad", "partae"]),
    #     ("model-00002-of-00003.safetensors", ["partaa", "partab", "partac", "partad", "partae"]),
    #     ("model-00003-of-00003.safetensors", ["partaa", "partab", "partac"]),
    # ]

    # Download small files
    for file_name in files_to_download:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            url = f"{release_url}/{file_name}"
            logger.info(f"Downloading {file_name} from {url}")
            response = requests.get(url, stream=True)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # Download and reassemble sharded .safetensors files
    for shard_name, parts in safetensors_shards:
        shard_path = os.path.join(model_dir, shard_name)
        if not os.path.exists(shard_path):
            part_paths = []
            for part in parts:
                part_name = f"{shard_name}.{part}"
                part_path = os.path.join(model_dir, part_name)
                url = f"{release_url}/{part_name}"
                logger.info(f"Downloading {part_name} from {url}")
                response = requests.get(url, stream=True)
                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                part_paths.append(part_path)

            # Reassemble the shard
            with open(shard_path, "wb") as f:
                for part_path in part_paths:
                    with open(part_path, "rb") as pf:
                        f.write(pf.read())
                    os.remove(part_path)  # Clean up part file

    return model_dir

# Load model and tokenizer
@st.cache_resource
def load_model():
    adapter_model_name = "woyeso/fine_tuned_llama_3_2_assignment_grader"
    hf_token = os.getenv("HF_TOKEN")
    
    # Use unsloth/Llama-3.2-1B-Instruct to reduce memory usage (comment out to revert to 3B model)
    base_model_name = "unsloth/Llama-3.2-1B-Instruct"
    # For unsloth/Llama-3.2-3B-Instruct (uncomment if using 3B model with upgraded hardware)
    # base_model_name = "unsloth/Llama-3.2-3B-Instruct"
    
    model_dir = download_model_weights()
    
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_model_name,
        token=hf_token if hf_token else None
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
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
uploaded_file = st.file_uploader("Upload PDF/DOCX", type QUIET["pdf", "docx"])
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