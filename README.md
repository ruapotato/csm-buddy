# Voice Chatbot Setup Instructions

This guide will help you set up the voice chatbot system that uses Whisper for speech recognition, Ollama Mistral for language processing, and CSM for speech synthesis.

## Prerequisites

1. Python 3.8+ installed
2. Ollama installed and running locally (https://ollama.com/)
3. Git
4. CUDA-compatible GPU recommended (for faster processing)

## Step 1: Install Ollama and Pull Mistral Model

1. Install Ollama from https://ollama.com/
2. Pull the Mistral model:
   ```
   ollama pull mistral
   ```
3. Start the Ollama server:
   ```
   ollama serve
   ```

## Step 2: Set Up the CSM Repository

1. Clone the CSM repository:
   ```
   git clone https://github.com/ruapotato/csm-buddy
   cd csm-buddy
   git submodule update --init --recursive

   ```

2. Create a virtual environment and install CSM dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Step 3: Download Whisper Model

The script will automatically download the Whisper model the first time you run it, but you can pre-download it:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Download the model (this will cache it for future use)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
```

## Step 4: Run the Voice Chatbot

1. Make sure Ollama is running with the Mistral model available
2. Make sure you have a working microphone
3. Run the voice chatbot script:
   ```
   python main.py
   Press enter when asked - talk - press enter when done
   
   python main_streaming.py (For streamed replys)
   ```

## Usage

1. The system will greet you when it starts
2. Press Enter to start recording your voice
3. Speak your message
4. Press Enter again to stop recording
5. The system will:
   - Show "LISTENING" debug when recording
   - Show "TRANSCRIBING" debug when processing your speech
   - Show "THINKING" debug when querying Ollama
   - Show "SPEAKING" debug when generating and playing the response
6. To exit, say "goodbye" or press Ctrl+C

## Troubleshooting

- **Audio recording issues**: Make sure your microphone is properly connected and configured as the default input device
- **Ollama connection error**: Ensure Ollama is running and the Mistral model is downloaded
- **CUDA/GPU errors**: If you encounter GPU-related errors, try modifying the script to use CPU only

