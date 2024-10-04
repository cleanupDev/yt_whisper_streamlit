import ollama


def get_summary_pipe(transcript_text: str):

    system_prompt = """You are an AI language model tasked with summarizing YouTube video transcripts.

        Instructions:

        Objective: Provide a clear and concise summary of the provided YouTube video transcript.

        Length & Format:

        The summary should be approximately 200-300 words.
        Present the summary in a well-structured paragraph format.
        Content Guidelines:

        Capture the main ideas, key points, and essential arguments presented in the video.
        Exclude minor details, repetitions, and any irrelevant information.
        If the video covers multiple topics, organize the summary to reflect this structure logically.
        Style & Tone:

        Maintain a neutral and informative tone.
        Ensure the summary is easy to read and understand, avoiding jargon unless necessary.
        Additional Instructions:

        Do not include timestamps, speaker labels, or metadata from the transcript.
        Ensure the summary is original and does not copy phrases verbatim from the transcript unless quoting is necessary for clarity.
        Quality Assurance:

        Verify that the summary accurately reflects the content and intent of the original video.
        Ensure coherence and logical flow throughout the summary."""

    user_prompt = "Summarize the following transcript from a youtube video: '"
    user_prompt += transcript_text

    return ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )


def ollama_stream_wrapper(pipe):
    for response in pipe:
        yield response["message"]["content"]
