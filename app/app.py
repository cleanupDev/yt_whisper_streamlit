from whisper_pipeline import get_whisper_pipe
from summary_pipeline import get_summary_pipe, ollama_stream_wrapper

import io as bytesio

import streamlit as st
from pytubefix import YouTube


status_display = st.empty()
summary_display = st.container()

sidebar_container_input = st.sidebar.container()
sidebar_container_output = st.sidebar.container()


status_display.title("YouTube Video Summarizer")


with sidebar_container_input:
    st.sidebar.text("YouTube URL")

    youtube_URL = st.sidebar.text_input(
        label="Input youtube video to summarize...",
        placeholder="https://youtube.com/watch?v=xxxxxxx",
        disabled=False,
        key="youtube_URL",
        on_change=lambda: summarize_main(),
    )


def summarize_main():
    with sidebar_container_output:
        with st.spinner(text="Please wait..."):
            youtube_URL = st.session_state.youtube_URL
            yt = YouTube(url=youtube_URL)
            st.write(yt.title)
            st.image(image=yt.thumbnail_url)

    with status_display.status("Summarizing video...", expanded=True) as status:
        st.write("Downloading audio...")
        audio_stream = yt.streams.get_audio_only()
        buffer = bytesio.BytesIO()
        audio_stream.stream_to_buffer(buffer)

        st.write("Starting whisper model...")
        pipe = get_whisper_pipe()

        st.write("Transcribing audio...")
        transcript = pipe(buffer.getvalue())
        transcript_result = transcript["text"]

        st.write("Summarizing transcript...")

        summary_display.write_stream(
            ollama_stream_wrapper(get_summary_pipe(transcript_result))
        )

        status.update(
            label="Video summarization complete!", state="complete", expanded=False
        )
