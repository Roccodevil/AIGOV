import os
import base64
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pdf2image import convert_from_path
from pypdf import PdfReader
from state import LegalGraphState

# Load environment variables
load_dotenv()

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_document(state: LegalGraphState):
    # 1. Check if raw text was passed directly
    if state.get("raw_text") and not state.get("file_path"):
        print("---RECEIVED DIRECT TEXT INPUT---")
        return {"raw_text": state["raw_text"], "current_step": "summarization"}
        
    file_path = state.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        raise ValueError("No valid file path or raw text provided.")

    # 2. Handle Plain Text Files
    if file_path.lower().endswith('.txt'):
        print("---READING PLAIN TEXT FILE---")
        with open(file_path, 'r', encoding='utf-8') as f:
            extracted_text = f.read().strip()
        return {"raw_text": extracted_text, "current_step": "summarization"}

    # 3. Handle PDFs (Try PyPDF first, fallback to OCR if empty/scanned)
    if file_path.lower().endswith('.pdf'):
        print("---ATTEMPTING FAST TEXT EXTRACTION VIA PYPDF---")
        try:
            reader = PdfReader(file_path)
            extracted_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
            
            # If PyPDF finds a meaningful amount of text, return it immediately
            if len(extracted_text.strip()) > 50:
                print("---SUCCESSFULLY EXTRACTED TEXT VIA PYPDF---")
                return {"raw_text": extracted_text.strip(), "current_step": "summarization"}
            else:
                print("---PYPDF FOUND NO TEXT (LIKELY SCANNED). FALLING BACK TO OCR---")
        except Exception as e:
            print(f"---PYPDF ERROR: {e}. FALLING BACK TO OCR---")

    # 4. Fallback: Hugging Face Vision OCR (For Images and Scanned PDFs)
    print("---EXTRACTING TEXT VIA HUGGING FACE VISION API---")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Please check your .env file.")
        
    client = InferenceClient(token=hf_token)
    extracted_text = ""
    image_paths = []

    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_paths.append(file_path)
    elif file_path.lower().endswith('.pdf'):
        # Convert the scanned PDF to images
        pages = convert_from_path(file_path, dpi=150)
        for i, page in enumerate(pages):
            temp_path = f"temp_page_{i}.jpg"
            page.save(temp_path, 'JPEG')
            image_paths.append(temp_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    # Process images through the Vision Model
    for img_path in image_paths:
        img_b64 = encode_image_to_base64(img_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a legal OCR system. Extract all readable text from this document image exactly as written. Do not add any commentary."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ]
        
        try:
            response = client.chat_completion(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                messages=messages,
                max_tokens=1500
            )
            extracted_text += response.choices[0].message.content + "\n\n"
        except Exception as e:
            extracted_text += f"[Error processing image: {e}]\n"
        finally:
            # Clean up generated image pages from PDFs
            if file_path.lower().endswith('.pdf') and os.path.exists(img_path):
                os.remove(img_path)

    return {"raw_text": extracted_text.strip(), "current_step": "summarization"}