import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Load your trained model and tokenizer
tokenizer = BartTokenizer.from_pretrained('./final_model_summarization')
model = BartForConditionalGeneration.from_pretrained('./final_model_summarization')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_summary(text, tokenizer, model, device):
    # Preprocess and tokenize the text
    text = clean_text(text)
    text = handle_special_content(text) 
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit user interface
st.title('Text Summarizer')
user_input = st.text_area("Enter the text you want to summarize", "Type Here...")
if st.button('Summarize'):
    summary = generate_summary(user_input, tokenizer, model, device)
    st.text_area("Summary", summary, height=250)
