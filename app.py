# Imports
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import io
from deep_translator import GoogleTranslator
from gtts import gTTS  # For multilingual text-to-speech
import uuid
import hashlib
import mysql.connector

# Database Connection
def create_connection():
    return mysql.connector.connect(
        host="localhost",  # Adjust if necessary
        user="root",
        password="KkA@19012005",
        database="ka"
    )

# Password Hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authentication Functions
def signup_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    try:
        cursor.execute("INSERT INTO Users (username, password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        st.success("Signup successful! Please log in.")
    except mysql.connector.Error as e:
        st.error(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

def login_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute("SELECT * FROM Users WHERE username = %s AND password = %s", (username, hashed_password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

# Streamlit App Title
st.title("Image to Story Generator with Authentication")

# Authentication State
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None

# Authentication Logic
if not st.session_state["logged_in"]:
    action = st.sidebar.radio("Choose Action", ["Login", "Signup"])
    if action == "Signup":
        st.subheader("Signup")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("Signup"):
            if password == confirm_password:
                signup_user(username, password)
            else:
                st.error("Passwords do not match.")
    elif action == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid credentials.")
else:
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None

    # After login -> Allow access to story generation
    st.write("Generate a story and audio from an image and translate it to your preferred language!")

    @st.cache_resource
    def load_blip_model():
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model

    @st.cache_resource
    def load_gpt2_model():
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        return tokenizer, model

    # Load models
    processor, blip_model = load_blip_model()
    gpt2_tokenizer, gpt2_model = load_gpt2_model()

    # Function to generate audio
    def generate_audio(story, language="en"):
        audio_file = f"audio_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=story, lang=language, slow=False)
        tts.save(audio_file)
        return audio_file

    # User Input Section
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    selected_language = st.selectbox(
        "Select Language",
        ["en", "es", "fr", "de", "hi", "zh-cn"],
        index=0
    )

    story_style = st.selectbox(
        "Select Story Style",
        ["Adventure", "Children", "Professional", "Emotional"],
        index=0
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate Caption and Story Button
        if st.button("Generate Story"):
            with st.spinner("Processing image and generating story..."):
                # Generate caption using BLIP
                inputs = processor(images=image, return_tensors="pt")
                generated_ids = blip_model.generate(
                    **inputs, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True
                )
                initial_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Modify the prompt and generation parameters based on the selected story style
                style_prompts = {
                    "Adventure": f"Write an adventurous and exciting story about {initial_caption}. Make it thrilling and action-packed.",
                    "Children": f"Write a short and simple children's story about {initial_caption}. Make it imaginative and fun.",
                    "Professional": f"Write a formal and professional story about {initial_caption}. Make it informative and structured.",
                    "Emotional": f"Write an emotional and heartfelt story about {initial_caption}. Make it touching and personal.",
                }
                tone_settings = {
                    "Adventure": {"temperature": 0.9, "top_k": 50, "top_p": 0.85},
                    "Children": {"temperature": 1.0, "top_k": 40, "top_p": 0.90},
                    "Professional": {"temperature": 0.7, "top_k": 30, "top_p": 0.80},
                    "Emotional": {"temperature": 0.8, "top_k": 50, "top_p": 0.85},
                }
                input_text = style_prompts[story_style]
                tone = tone_settings[story_style]

                # Generate story using GPT-2
                input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
                output = gpt2_model.generate(
                    input_ids,
                    max_length=500,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    num_beams=5,
                    temperature=tone["temperature"],
                    top_k=tone["top_k"],
                    top_p=tone["top_p"],
                    early_stopping=True,
                )
                story = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

                # Translate story if needed
                if selected_language != "en":
                    translated_text = GoogleTranslator(source="auto", target=selected_language).translate(story)
                    story = translated_text

                # Generate audio in the selected language
                audio_file = generate_audio(story, language=selected_language)

                # Display results
                st.subheader("Generated Caption")
                st.write(initial_caption)

                st.subheader("Generated Story")
                st.text_area("Story", value=story, height=200)

                st.subheader("Audio File")
                st.audio(audio_file, format="audio/mp3")
