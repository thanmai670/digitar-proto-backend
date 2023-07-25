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
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

dg_client = Deepgram(os.getenv("DEEPGRAM_API_KEY"))

templates = Jinja2Templates(directory="templates")

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


async def process_audio(fast_socket: WebSocket):
    async def get_transcript(data: Dict) -> None:
        if "channel" in data:
            transcript = data["channel"]["alternatives"][0]["transcript"]

            if transcript:
                # Generate response using GPT-2
                response = generate_response(transcript)
                await fast_socket.send_text(
                    '{"input_text": "'
                    + transcript
                    + '", "response": "'
                    + response
                    + '"}'
                )

    deepgram_socket = await connect_to_deepgram(get_transcript)

    return deepgram_socket


async def connect_to_deepgram(transcript_received_handler: Callable[[Dict], None]):
    try:
        socket = await dg_client.transcription.live(
            {
                "punctuate": True,
                "interim_results": False,
                "utterances": True,
                "model": "nova",
                "numerals": True,
            }
        )
        socket.registerHandler(
            socket.event.CLOSE, lambda c: print(f"Connection closed with code {c}.")
        )
        socket.registerHandler(
            socket.event.TRANSCRIPT_RECEIVED, transcript_received_handler
        )

        return socket
    except Exception as e:
        raise Exception(f"Could not open socket: {e}")


def generate_response(input_text):
    # GPT-3 generation process remains the same...
    gpt3_api_key = "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"
    openai.api_key = gpt3_api_key

    # Call the GPT-3 API to generate a response
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can choose a different engine from the OpenAI API docs
        prompt=input_text,
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


@app.get("/", response_class=HTMLResponse)
def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            input_data = json.loads(data)

            print(f"Received data: {input_data}")  # Print received data

            if "input" in input_data:
                response_text, response_audio = generate_response(input_data["input"])
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
