import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="StyleSense", layout="wide")
st.title("🎨 StyleSense - Fashion Mood Analyzer & Matcher")

uploaded_image = st.file_uploader("📸 Upload Outfit Photo", type=["png", "jpg", "jpeg"])
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Outfit", use_column_width=True)

    with st.spinner("🧠 Analyzing outfit..."):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.subheader("📝 Outfit Description:")
    st.write(caption)

    prompt = PromptTemplate.from_template(
        "The outfit is described as: {caption}. Suggest complementary fashion accessories, shoes, or jackets. Also generate a stylish Instagram caption and 5 trending hashtags."
    )
    full_prompt = prompt.format(caption=caption)
    llm = OpenAI(temperature=0.7)
    result = llm(full_prompt)

    st.subheader("👗 Fashion Suggestions & Caption:")
    st.text_area("Output", result, height=300)

    st.success("✨ Style matched successfully!")
