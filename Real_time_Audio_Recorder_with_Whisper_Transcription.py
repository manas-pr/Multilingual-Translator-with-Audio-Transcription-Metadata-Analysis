
import streamlit as st
from st_audiorec import st_audiorec
import whisper
from translate import Translator
import os
from datetime import datetime
from io import BytesIO

# Load Whisper Model
model = whisper.load_model("base")

# Translation Function
def translate_text(text, target_lang):
    translator = Translator(to_lang=target_lang)
    try:
        translation = translator.translate(text)
        return translation
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI Enhancements
st.set_page_config(page_title="🎧 Enhanced Audio Recorder")
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''', unsafe_allow_html=True)
st.markdown('''<style>.stAudio {height: 45px;}</style>''', unsafe_allow_html=True)

# Main Function
def audiorec_demo_app():
    st.title('🎧 Audio Recorder with Real-Time Whisper Transcription & Translation')
    st.markdown('Developed with ❤️ by [Manas Pratim](https://github.com/manas-pr)')

    # Audio Recorder and File Upload
    st.subheader("🎤 Record or Upload Audio")
    wav_audio_data = st_audiorec()
    uploaded_audio = st.file_uploader("📂 Upload Audio File (WAV format)", type=["wav"])

    # Handle Audio Data (Recorder or Uploaded)
    if uploaded_audio is not None:
        audio_data = uploaded_audio.read()
        filename = uploaded_audio.name
    elif wav_audio_data is not None:
        audio_data = wav_audio_data
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    # Proceed if audio is available
    if 'audio_data' in locals():
        st.audio(audio_data, format='audio/wav')

        # Save audio data to a file
        with open(filename, "wb") as f:
            f.write(audio_data)

        # Transcription Button
        if st.button("📝 Transcribe Audio"):
            with st.spinner("🧠 Step 1: Loading Audio..."):
                st.info("✅ Audio loaded successfully.")

            # Real-Time Transcription Progress
            with st.spinner("🧠 Step 2: Transcribing Audio..."):
                transcription_data = model.transcribe(filename)
                transcription_text = transcription_data['text']

                # Word-by-Word Progress in Single Line
                st.subheader("🔎 Real-Time Transcription Progress")
                placeholder = st.empty()  # Dynamic display
                full_text = ""
                for word in transcription_text.split():
                    full_text += word + " "
                    placeholder.markdown(f"**{full_text.strip()}**")

                # Display Final Transcription
                st.subheader("📝 Final Transcription (English)")
                st.success(transcription_text)

                # Language Selection for Translation
                st.subheader("🌐 Choose Translation Language")
                translation_lang = st.radio(
                    "Select Language:",
                    options=["Assamese", "Hindi"],
                    horizontal=True
                )

                # Translation Button
                if st.button("🌐 Translate"):
                    with st.spinner(f"🔄 Translating to {translation_lang}..."):
                        target_lang_code = "as" if translation_lang == "Assamese" else "hi"
                        translated_text = translate_text(transcription_text, target_lang_code)
                        st.subheader(f"📝 Translation ({translation_lang})")
                        st.success(translated_text)
                        print(f"Translated Text ({translation_lang}): {translated_text}")

            # Transcription Accuracy Details
            st.subheader("📊 Transcription Accuracy Details")
            st.write(f"🔹 **Segments Processed:** {len(transcription_data['segments'])}")
            st.write(f"🔹 **Language Detected:** {transcription_data['language']}")

            # Detailed Segment Information
            for i, segment in enumerate(transcription_data['segments']):
                st.write(f"**Segment {i + 1}:**")
                st.write(f"🗣️ **Text:** {segment['text']}")
                st.write(f"✅ **Confidence:** {segment['avg_logprob']:.2f}")
                st.write(f"💇 **No Speech Probability:** {segment['no_speech_prob']:.2%}")
                st.write(f"🕒 **Start Time:** {segment['start']:.2f}s - **End Time:** {segment['end']:.2f}s")
                st.markdown("---")

        # Download Button
        st.download_button(
            label="⬇️ Download Audio",
            data=audio_data,
            file_name=filename,
            mime="audio/wav"
        )

if __name__ == '__main__':
    audiorec_demo_app()
