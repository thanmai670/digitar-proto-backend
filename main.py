import json
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Callable
from deepgram import Deepgram
from dotenv import load_dotenv
import openai
import boto3
import base64
import riva.client
from pydantic import BaseModel
from copy import deepcopy
import wave
import riva.client.audio_io
import IPython

load_dotenv()

app = FastAPI()


auth = riva.client.Auth(uri="3.84.243.225:50051")

# dg_client = Deepgram(os.getenv("DEEPGRAM_API_KEY"))

templates = Jinja2Templates(directory="templates")


def generate_response(input_text):
    # GPT-3 generation process remains the same...
    gpt3_api_key = "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"
    openai.api_key = gpt3_api_key

    # Call the GPT-3 API to generate a response
    response = openai.ChatCompletion.create(
        engine="text-davinci-003",
        max_tokens=100,
        temperature=0.2,
    )

    generated_text = response.choices[0].text.strip()
    # Initialize boto3 client for Polly
    polly_client = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION_NAME"),
    ).client("polly")

    # Request speech synthesis
    response = polly_client.synthesize_speech(
        Text=generated_text, OutputFormat="mp3", VoiceId="Joanna"
    )
    # Convert the response to base64 string so it can be sent via WebSocket
    audio_stream = base64.b64encode(response["AudioStream"].read()).decode("utf-8")

    # Create a data URL for the audio file
    audio_url = "data:audio/mp3;base64," + audio_stream

    # Return the response and audio url
    return generated_text, audio_url


# @app.get("/", response_class=HTMLResponse)
# def get(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    asr_service = riva.client.ASRService(auth)
    offline_config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
        audio_channel_count=1,
    )
    streaming_config = riva.client.StreamingRecognitionConfig(
        config=deepcopy(offline_config), interim_results=True
    )

    try:
        while True:
            audio_data = await websocket.receive_bytes()
            # input_data = json.loads(data)

            print(f"Received data: {audio_data}")  # Print received data

            audio_chunk_iterator = riva.client.AudioChunkBytesIterator(audio_data, 4800)
            response_generator = asr_service.streaming_response_generator(
                audio_chunk_iterator, streaming_config
            )
            for streaming_response in response_generator:
                result_text = riva.client.get_transcript_from_streaming_response(
                    streaming_response
                )

                response_text, response_audio = generate_response(result_text)
                print(
                    f"Response generated: {response_text}"
                )  # Print generated response

                await websocket.send_text(
                    json.dumps(
                        {
                            "response_text": response_text,
                            "response_audio": response_audio,
                        }
                    )
                )

    except Exception as e:
        print(f"Error: {e}")  # Print error
        raise Exception(f"Could not process audio: {e}")
    finally:
        await websocket.close()


# asr_service = riva.client.ASRService(auth)
# offline_config = riva.client.RecognitionConfig(
#     encoding=riva.client.AudioEncoding.LINEAR_PCM,
#     max_alternatives=1,
#     enable_automatic_punctuation=True,
#     verbatim_transcripts=False,
#     audio_channel_count=1,
# )
# streaming_config = riva.client.StreamingRecognitionConfig(
#     config=deepcopy(offline_config), interim_results=True
# )

# my_wav_file = "speech1.wav"
# riva.client.add_audio_file_specs_to_config(offline_config, my_wav_file)
# riva.client.add_audio_file_specs_to_config(streaming_config, my_wav_file)

# boosted_lm_words = ["AntiBERTa", "ABlooper"]
# boosted_lm_score = 20.0
# riva.client.add_word_boosting_to_config(
#     offline_config, boosted_lm_words, boosted_lm_score
# )
# riva.client.add_word_boosting_to_config(
#     streaming_config, boosted_lm_words, boosted_lm_score
# )

# wav_parameters = riva.client.get_wav_file_parameters(my_wav_file)
# # correponds to 1 second of audio
# chunk_size = wav_parameters["framerate"]
# with riva.client.AudioChunkFileIterator(
#     my_wav_file,
#     chunk_size,
#     delay_callback=riva.client.sleep_audio_length,
# ) as audio_chunk_iterator:
#     for i, chunk in enumerate(audio_chunk_iterator):
#         print(i, len(chunk))

# audio_chunk_iterator = riva.client.AudioChunkFileIterator(my_wav_file, 4800)
# response_generator = asr_service.streaming_response_generator(
#     audio_chunk_iterator, streaming_config
# )
# streaming_response = next(response_generator)

# riva.client.print_streaming(response_generator, additional_info="time")

# audio_chunk_iterator = riva.client.AudioChunkFileIterator(
#     my_wav_file, 4800, riva.client.sleep_audio_length
# )
# response_generator = asr_service.streaming_response_generator(
#     audio_chunk_iterator, streaming_config
# )
# riva.client.print_streaming(response_generator, show_intermediate=True)

tts_service = riva.client.SpeechSynthesisService(auth)
language_code = "en-US"
sample_rate_hz = 16000
nchannels = 1
sampwidth = 2
text = (
    "The United States of America, commonly known as the United States or America, "
    "is a country primarily located in North America. It consists of 50 states, "
    "a federal district, five major unincorporated territories, 326 Indian reservations, "
    "and nine minor outlying islands."
)
responses = tts_service.synthesize_online(
    text, language_code=language_code, sample_rate_hz=sample_rate_hz
)

streaming_audio = b""
for resp in responses:
    streaming_audio += resp.audio

streaming_output_file = "my_streaming_synthesized_speech.wav"
with wave.open(streaming_output_file, "wb") as out_f:
    out_f.setnchannels(nchannels)
    out_f.setsampwidth(sampwidth)
    out_f.setframerate(sample_rate_hz)
    out_f.writeframesraw(streaming_audio)


IPython.display.Audio(streaming_output_file)
