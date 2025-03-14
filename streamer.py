#!/usr/bin/env python3
"""
Standalone Streaming TTS implementation for CSM
"""

# Standard library imports
import os
import time
import sys
import threading
import queue
import tempfile

# Third-party imports - make sure all are explicit
import numpy as np
import torch
import torchaudio
import sounddevice as sd
from huggingface_hub import hf_hub_download

# Import generator - this is a relative import from the csm directory
# This import assumes the script is in the same directory as the csm module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csm.generator import load_csm_1b, Segment

# Simple colored output without dependencies
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

def print_debug(message, category="DEBUG", color=CYAN):
    """Print a colored debug message with timestamp"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"{color}[{timestamp}] {category}: {message}{RESET}")

def print_status(status, description=""):
    """Print a very visible status message"""
    colors = {
        "LISTENING": GREEN,
        "TRANSCRIBING": YELLOW,
        "THINKING": BLUE,
        "SPEAKING": MAGENTA,
        "ERROR": RED
    }
    color = colors.get(status, CYAN)
    
    # Print a visually distinct status message
    border = "=" * 50
    print(f"\n{color}{border}")
    print(f"  {status}: {description}")
    print(f"{border}{RESET}\n")

class StreamingCSMGenerator:
    """Streaming TTS implementation for CSM speech synthesis"""
    
    def __init__(self, model_path=None):
        print_debug("Initializing optimized streaming CSM speech model...", "TTS")
        
        if model_path is None:
            print_debug("Downloading CSM model from Hugging Face", "TTS")
            model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
        
        print_debug(f"Loading CSM model from: {model_path}", "TTS")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Standard model loading
            self.generator = load_csm_1b(model_path, device)
            print_debug(f"CSM model loaded successfully on {device}", "TTS")
        except Exception as e:
            print_status("ERROR", f"Failed to load CSM model: {str(e)}")
            raise
        
        self.conversation_segments = []
        self.speaker_ids = [0, 1]  # 0 for assistant, 1 for user
        self.sample_rate = self.generator.sample_rate
        self.audio_queue = queue.Queue()
        self.is_generating = False
        self.playback_thread = None
        print_debug("Optimized streaming speech generator initialized", "TTS")
    
    def generate_speech_streaming(self, text, output_path="output.wav"):
        """Generate speech in a streaming fashion and play it as it's generated"""
        print_status("SPEAKING", "Generating speech (streaming)...")
        print_debug(f"Generating speech for text ({len(text)} chars)", "TTS")
        
        # Clear any existing audio in the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        self.is_generating = True
        
        # Start playback thread if not already running
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(
                target=self._streaming_playback_worker, 
                daemon=True
            )
            self.playback_thread.start()
        
        # Start generation thread
        generation_thread = threading.Thread(
            target=self._generate_speech_worker,
            args=(text, output_path),
            daemon=True
        )
        generation_thread.start()
        
        return output_path
    
    def _generate_speech_worker(self, text, output_path):
        """Worker thread that generates speech and adds chunks to the queue"""
        try:
            print_debug("Starting optimized speech generation thread", "TTS")
            start_time = time.time()
            
            # Use aggressive chunking for faster streaming
            chunks = self._chunk_text(text)
            full_audio = None
            
            for i, chunk in enumerate(chunks):
                chunk_start = time.time()
                print_debug(f"Generating chunk {i+1}/{len(chunks)}: '{chunk}'", "TTS")
                
                # Generate audio with torch.inference_mode for speed
                with torch.inference_mode():
                    audio = self.generator.generate(
                        text=chunk,
                        speaker=self.speaker_ids[0],
                        context=self.conversation_segments,
                        max_audio_length_ms=3000,  # Short segment max
                        temperature=1.0,  # Higher temp = faster but less accurate
                        topk=10,  # Lower topk = faster sampling
                    )
                
                chunk_time = time.time() - chunk_start
                print_debug(f"Chunk generated in {chunk_time:.2f}s", "TTS")
                
                # Add to queue for playback
                self.audio_queue.put(audio)
                
                # Accumulate for full audio
                if full_audio is None:
                    full_audio = audio
                else:
                    full_audio = torch.cat([full_audio, audio])
            
            # Signal end of generation
            self.audio_queue.put(None)  # None as sentinel to indicate completion
            
            # Save complete audio
            torchaudio.save(output_path, full_audio.unsqueeze(0).cpu(), self.sample_rate)
            
            # Only create a segment for the full utterance to save memory
            new_segment = Segment(
                text=text,
                speaker=self.speaker_ids[0],
                audio=full_audio
            )
            
            # Add to conversation history (limit to prevent context getting too large)
            self.conversation_segments.append(new_segment)
            if len(self.conversation_segments) > 3:  # Keep just 3 segments max for speed
                self.conversation_segments = self.conversation_segments[-3:]
            
            processing_time = time.time() - start_time
            print_debug(f"Speech generation completed in {processing_time:.2f}s", "TTS")
            print_debug(f"Complete audio saved to: {output_path}", "TTS")
            
        except Exception as e:
            error_msg = f"Error in speech generation: {str(e)}"
            print_status("ERROR", error_msg)
            import traceback
            traceback.print_exc()
            self.audio_queue.put(None)  # Signal completion even on error
        finally:
            self.is_generating = False
    
    def _streaming_playback_worker(self):
        """Worker thread that plays audio chunks from the queue"""
        print_debug("Starting audio streaming playback thread", "AUDIO")
        
        # This will run until a None is received and no generation is happening
        while True:
            try:
                # Get next audio chunk with timeout
                chunk = self.audio_queue.get(timeout=0.5)
                
                # If we got None and generation is done, exit
                if chunk is None:
                    if not self.is_generating:
                        print_debug("End of streaming playback", "AUDIO")
                        break
                    continue
                
                # Play this chunk
                self._play_audio_chunk(chunk)
                
            except queue.Empty:
                # No new chunks yet, but generation might still be ongoing
                if not self.is_generating:
                    # If no more chunks and generation is done, exit
                    print_debug("No more audio chunks and generation complete", "AUDIO")
                    break
            except Exception as e:
                print_debug(f"Error in audio playback: {str(e)}", "AUDIO", RED)
                break
    
    def _play_audio_chunk(self, audio_tensor):
        """Play a single audio chunk"""
        try:
            # Convert to numpy for sounddevice
            audio_np = audio_tensor.cpu().numpy()
            
            # Play audio
            sd.play(audio_np, self.sample_rate)
            sd.wait()
        except Exception as e:
            print_debug(f"Error playing audio chunk: {str(e)}", "AUDIO", RED)
    
    def _chunk_text(self, text):
        """Split text into reasonable chunks for streaming"""
        # If text is very short, don't chunk it
        if len(text) < 30:
            return [text]
        
        chunks = []
        current_chunk = ""
        words = text.split()
        
        # Break text into chunks of 3-5 words
        chunk_size = min(5, max(3, len(words) // 10))  # Dynamic chunk size based on text length
        
        for i, word in enumerate(words):
            current_chunk += word + " "
            
            # Break at regular intervals or punctuation
            if (i + 1) % chunk_size == 0 or any(p in word for p in ['.', '!', '?', ',', ';', ':']):
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        # Add any remaining text
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def add_user_segment(self, text, audio_path):
        """Add a user utterance to the conversation history"""
        print_debug("Adding user utterance to conversation history", "TTS")
        try:
            # Load audio
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != self.generator.sample_rate:
                print_debug(f"Resampling from {sample_rate}Hz to {self.generator.sample_rate}Hz", "TTS")
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor.squeeze(0), 
                    orig_freq=sample_rate, 
                    new_freq=self.generator.sample_rate
                )
            else:
                audio_tensor = audio_tensor.squeeze(0)
            
            # Create segment
            user_segment = Segment(
                text=text,
                speaker=self.speaker_ids[1],  # User is speaker 1
                audio=audio_tensor
            )
            
            # Add to conversation history
            self.conversation_segments.append(user_segment)
            if len(self.conversation_segments) > 3:  # Keep just 3 segments max for speed
                self.conversation_segments = self.conversation_segments[-3:]
            
            print_debug(f"Added user segment", "TTS")
        
        except Exception as e:
            print_debug(f"Error adding user segment: {str(e)}", "TTS", RED)

# This allows the module to be run directly for testing
if __name__ == "__main__":
    # Test the streaming speech generator
    tts = StreamingCSMGenerator()
    tts.generate_speech_streaming("This is a test of the streaming speech generator. Is it working properly?")
    
    # Wait for playback to finish
    time.sleep(5)
    print_debug("Test complete", "SYSTEM")
