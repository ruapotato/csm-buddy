#!/usr/bin/env python3
"""
Main script for Voice Chatbot with Streaming TTS
"""

import os
import time
import sys
import threading
import tempfile
import wave
import torch
import torchaudio
import pyaudio
import requests
import json
import traceback
from transformers import WhisperProcessor, WhisperForConditionalGeneration
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csm'))

# Import our streaming TTS implementation
from streamer import StreamingCSMGenerator, print_debug, print_status

# Simple colored output without dependencies
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = []
        self.is_recording = False
        self.pyaudio = pyaudio.PyAudio()
        
    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        print_status("LISTENING", "Recording your voice...")
        print_debug("Opening audio stream", "AUDIO")
        
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.callback
        )
        
        self.frames = []
        self.is_recording = True
        
        # Visual feedback of recording
        print_debug("Recording started - press Enter when done speaking", "AUDIO")
        
        # Audio level monitoring in a separate thread
        def monitor_audio():
            indicators = ["|", "/", "-", "\\"]
            idx = 0
            while self.is_recording:
                if self.frames and len(self.frames) > 0:
                    # Simple visualization that recording is happening
                    sys.stdout.write(f"\r{GREEN}Recording {indicators[idx]} {RESET}")
                    sys.stdout.flush()
                    idx = (idx + 1) % len(indicators)
                time.sleep(0.2)
            sys.stdout.write("\r" + " " * 20 + "\r")  # Clear the indicator line
            sys.stdout.flush()
            
        threading.Thread(target=monitor_audio, daemon=True).start()
    
    def stop_recording(self):
        print_debug("Stopping recording", "AUDIO")
        self.is_recording = False
        
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        
        # Save the recorded audio to a temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        print_debug(f"Saving audio to {temp_file.name}", "AUDIO")
        
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        print_debug(f"Recorded {len(self.frames)} frames", "AUDIO")
        return temp_file.name
    
    def cleanup(self):
        print_debug("Cleaning up audio resources", "AUDIO")
        self.pyaudio.terminate()

class WhisperASR:
    def __init__(self, model_name="openai/whisper-small"):
        print_status("INITIALIZING", "Loading Whisper ASR model...")
        print_debug(f"Loading Whisper model: {model_name}", "ASR")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            print_debug("Moving Whisper model to GPU", "ASR")
            self.model = self.model.to("cuda")
        else:
            print_debug("Running Whisper on CPU (GPU not available)", "ASR")
    
    def transcribe(self, audio_path):
        print_status("TRANSCRIBING", "Converting speech to text...")
        print_debug(f"Processing audio file: {audio_path}", "ASR")
        
        # Load audio file
        audio_input, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != 16000:
            print_debug(f"Resampling from {sample_rate}Hz to 16000Hz", "ASR")
            audio_input = torchaudio.functional.resample(audio_input, sample_rate, 16000)
        
        # Convert to numpy
        audio_input = audio_input.squeeze().numpy()
        
        # Process with Whisper
        print_debug("Running Whisper inference...", "ASR")
        start_time = time.time()
        
        input_features = self.processor(
            audio_input, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features
        
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")
        
        # Generate transcription
        print_debug("Generating transcription...", "ASR")
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        processing_time = time.time() - start_time
        print_debug(f"Transcription completed in {processing_time:.2f}s", "ASR")
        print_debug(f"Transcript: '{transcription}'", "ASR")
        
        return transcription

class Ollama:
    def __init__(self, model="mistral", base_url="http://localhost:11434"):
        print_status("INITIALIZING", "Setting up Ollama LLM interface...")
        self.model = model
        self.base_url = base_url
        
        # Initialize conversation history with a system prompt for casual conversation
        self.conversation_history = [
            {
                "role": "system", 
                "content": "You are a friendly and casual assistant. Use conversational language, " +
                          "informal expressions, and respond in a relaxed way as if chatting with a friend. " +
                          "Keep responses BRIEF and engaging. For example, if someone says 'Hey!', " +
                          "you might respond with 'Yo! How's it going?' rather than formal greetings. The shorter the better!"
            }
        ]
        
        print_debug(f"Initialized Ollama LLM interface using model: {model}", "LLM")
    
    def get_response(self, text):
        print_status("THINKING", "Processing with language model...")
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": text})
        
        # Prepare the request
        data = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": False
        }
        
        print_debug(f"Sending request to Ollama model: {self.model}", "LLM")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps(data),
                timeout=60  # 1 minute timeout
            )
            response.raise_for_status()
            
            result = response.json()
            processing_time = time.time() - start_time
            print_debug(f"Received response in {processing_time:.2f}s", "LLM")
            
            # Extract response text
            response_text = result.get("message", {}).get("content", "")
            
            # Add assistant message to history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Log first 100 chars of response
            preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
            print_debug(f"Response: '{preview}'", "LLM")
            
            return response_text
        
        except requests.RequestException as e:
            error_msg = f"Error communicating with Ollama: {str(e)}"
            print_status("ERROR", error_msg)
            
            if hasattr(e, 'response') and e.response is not None:
                print_debug(f"Response status: {e.response.status_code}", "LLM", RED)
                print_debug(f"Response body: {e.response.text}", "LLM", RED)
            
            return "Oops! Something went wrong while I was trying to answer. Can you try again?"

def main():
    print_status("STARTING", "Voice Chatbot System (with Streaming TTS)")
    print_debug("Press Ctrl+C to exit", "SYSTEM")
    
    try:
        # Initialize components
        print_debug("Initializing ASR (Whisper)...", "SYSTEM")
        asr = WhisperASR()
        
        print_debug("Initializing LLM (Ollama Mistral)...", "SYSTEM")
        llm = Ollama()
        
        print_debug("Initializing Streaming TTS...", "SYSTEM")
        tts = StreamingCSMGenerator()
        
        print_debug("Initializing Audio Recorder...", "SYSTEM")
        recorder = AudioRecorder()
        
        print_status("READY", "Initialization complete")
        
        # Initial greeting
        greeting = "Hello! I'm your voice assistant with streaming TTS. How can I help you today?"
        print_debug(f"Assistant: {greeting}", "SYSTEM")
        
        # Use streaming TTS for the greeting
        tts.generate_speech_streaming(greeting)
        
        # Main conversation loop
        while True:
            print_status("NEW TURN", "Starting new conversation turn")
            print_debug("Press Enter to start speaking, then press Enter again when done.", "SYSTEM")
            input()
            
            # Start recording
            record_thread = threading.Thread(target=recorder.start_recording, daemon=True)
            record_thread.start()
            input()
            
            # Stop recording and get audio file
            audio_file = recorder.stop_recording()
            
            # Transcribe with Whisper
            transcription = asr.transcribe(audio_file)
            print_debug(f"User said: {transcription}", "SYSTEM")
            
            # Add user segment to conversation
            tts.add_user_segment(transcription, audio_file)
            
            # If user wants to exit
            if any(exit_phrase in transcription.lower() for exit_phrase in ["exit", "quit", "goodbye", "bye"]):
                farewell = "Goodbye! Have a great day!"
                print_debug(f"Assistant: {farewell}", "SYSTEM")
                tts.generate_speech_streaming(farewell)
                # Wait for playback to finish
                time.sleep(2)
                break
            
            # Get response from LLM
            response = llm.get_response(transcription)
            print_debug(f"Assistant: {response}", "SYSTEM")
            
            # Generate and play streaming speech
            tts.generate_speech_streaming(response)
    
    except KeyboardInterrupt:
        print_status("SHUTTING DOWN", "Interrupted by user")
    except Exception as e:
        print_status("ERROR", str(e))
        traceback.print_exc()
    finally:
        # Cleanup
        print_debug("Cleaning up resources...", "SYSTEM")
        if 'recorder' in locals():
            recorder.cleanup()
        print_debug("System shutdown complete", "SYSTEM")
        print_status("TERMINATED", "System shutdown complete")

if __name__ == "__main__":
    main()
