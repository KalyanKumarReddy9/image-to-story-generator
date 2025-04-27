import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid
import io

# Initialize models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Helper function to generate caption from image
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = blip_model.generate(
        **inputs, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True
    )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

# Helper function to generate a story from the caption
def generate_story(caption, story_style="Adventure"):
    style_prompts = {
        "Adventure": f"Write an adventurous story about {caption}.",
        "Children": f"Write a children's story about {caption}.",
        "Professional": f"Write a professional story about {caption}.",
        "Emotional": f"Write an emotional story about {caption}.",
    }
    input_text = style_prompts.get(story_style, style_prompts["Adventure"])  # Default to Adventure
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    output = gpt2_model.generate(
        input_ids,
        max_length=500,
        num_beams=5,
        no_repeat_ngram_size=2,
        temperature=0.9,
        top_k=50,
        top_p=0.85,
        early_stopping=True,
    )
    story = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# Helper function to translate the story
def translate_story(story, target_language):
    if target_language != "en":
        return GoogleTranslator(source="auto", target=target_language).translate(story)
    return story

# Helper function to generate audio of the story
def generate_audio(story, language="en"):
    audio_file = f"audio_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=story, lang=language, slow=False)
    tts.save(audio_file)
    return audio_file

# Streamlit Interface
def app():
    st.title("Story Generator")
    
    # File uploader for image
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    # Dropdown for selecting story style
    story_style = st.selectbox("Select Story Style", ["Adventure", "Children", "Professional", "Emotional"])

    # Dropdown for selecting target language
    target_language = st.selectbox("Select Target Language", ["en", "es", "fr", "de", "it"])

    if image_file is not None:
        # Display the uploaded image
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        try:
            # Generate caption from image
            caption = generate_caption(image)
            st.subheader("Generated Caption")
            st.write(caption)

            # Generate story from caption
            story = generate_story(caption, story_style)
            st.subheader("Generated Story")
            st.write(story)

            # Translate the story
            translated_story = translate_story(story, target_language)
            st.subheader("Translated Story")
            st.write(translated_story)

            # Generate audio from the story
            audio_file = generate_audio(translated_story, target_language)
            st.subheader("Generated Audio")
            st.audio(audio_file, format="audio/mp3")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app()
