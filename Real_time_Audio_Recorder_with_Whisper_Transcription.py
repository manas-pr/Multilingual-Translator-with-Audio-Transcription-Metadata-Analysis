import streamlit as st
from translate import Translator
import whisper
import edge_tts
import asyncio
import soundfile as sf
import tempfile  # For temporary file creation

# Load Whisper Model for Transcription
model = whisper.load_model("base")

# Function to translate text
def translate_text(text, target_lang):
    translator = Translator(to_lang=target_lang)
    try:
        translation = translator.translate(text)
        return translation
    except Exception as e:
        return f"Error: {e}"

# Function for TTS with safe parameters
async def text_to_speech(text, voice, speed="+0%"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        communicate = edge_tts.Communicate(text, voice=voice, rate=speed)
        await communicate.save(temp_audio.name)
        return temp_audio.name

# Function to Save Uploaded File
def save_uploaded_file(uploaded_file):
    file_path = f"uploaded_audio.{uploaded_file.type.split('/')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

# Function to Extract Audio Metadata
def get_audio_info(file_path):
    try:
        audio_data, sample_rate = sf.read(file_path)
        duration = len(audio_data) / sample_rate
        file_size = round((len(audio_data) * 2) / 1024, 2)  # File size in KB
        return duration, sample_rate, file_size
    except Exception as e:
        return 0, 0, 0

# Main Function
def audiorec_demo_app():
    st.title('🎧 English Translator with Audio Transcription & Translation')
    st.markdown('Developed with ❤️ by [Manas Pratim](https://github.com/manas-pr)')

    # File Upload
    st.subheader("📂 Upload Audio File (WAV format)")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

    # Handle Audio Data (Uploaded File Only)
    if uploaded_file:
        audio_path = save_uploaded_file(uploaded_file)

        # Display audio details
        duration, sample_rate, file_size = get_audio_info(audio_path)
        st.success(f"✅ Audio saved: {audio_path}")
        st.audio(audio_path)

        # Display metadata
        st.markdown(f"""
        **🕒 Duration:** {round(duration, 2)} seconds  
        **🎚️ Sample Rate:** {sample_rate} Hz  
        **📂 File Size:** {file_size} KB  
        """)

        # Transcription Process
        with st.spinner("🔄 Transcribing audio..."):
            transcription_data = model.transcribe(audio_path)
            transcription_text = transcription_data['text']
            st.subheader("📝 Transcribed Text (English)")
            st.success(transcription_text)

            # Transcription Accuracy Details
            st.subheader("📊 Transcription Accuracy Details")
            st.write(f"🔹 **Segments Processed:** {len(transcription_data['segments'])}")
            st.write(f"🔹 **Language Detected:** {transcription_data['language']}")

            # Detailed Segment Information
            for i, segment in enumerate(transcription_data['segments']):
                st.write(f"🗣️ **Text:** {segment['text']}")
                st.write(f"✅ **Confidence:** {segment['avg_logprob']:.2f}")
                st.write(f"🔇 **No Speech Probability:** {segment['no_speech_prob']:.2%}")
                st.write(f"🕒 **Start Time:** {segment['start']:.2f}s - **End Time:** {segment['end']:.2f}s")
                st.markdown("---")

            # Use transcribed text for translation
            english_text = transcription_text

    # Translation Language Selection
    translation_lang = st.selectbox(
        "🌐 Choose Translation Language:",
        options=["Assamese", "Hindi"]
    )

    # Speed Control Slider
    speed_options = st.select_slider(
        "🎯 Select Speaking Speed:",
        options=["-50%", "-25%", "+0%", "+25%", "+50%"],
        value="+0%"
    )

    # Voice Selection Dropdown
    voice_options = {
        "Assamese": ["bn-IN-TanishaaNeural", "bn-IN-BashkarNeural"],
        "Hindi": ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"]
    }
    selected_voice = st.selectbox("🎙️ Select Voice:", options=voice_options[translation_lang])

    # Button Layout
    col1, col2 = st.columns(2)
    with col1:
        translate_button = st.button("Translate")
    with col2:
        clear_button = st.button("Clear")

    # Translation Logic
    if translate_button:
        if english_text.strip():
            target_lang_code = "as" if translation_lang == "Assamese" else "hi"
            translated_text = translate_text(english_text, target_lang_code)
            st.success(f"**Translated Text ({translation_lang}):** {translated_text}")
            
            # TTS with improved voice selection and error handling
            try:
                audio_file = asyncio.run(text_to_speech(translated_text, selected_voice, speed_options))
                st.audio(audio_file, format="audio/mp3", start_time=0)
            except Exception as e:
                st.error(f"⚠️ TTS Error: {e}")
        else:
            st.warning("⚠️ Please enter some text for translation or upload an audio file.")

    # Clear Button Logic
    if clear_button:
        st.experimental_rerun()

if __name__ == '__main__':
    audiorec_demo_app()
