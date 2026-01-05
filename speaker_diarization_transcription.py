import os
import tempfile
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
import whisper
from dotenv import load_dotenv
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

def initialize_diarization_pipeline():
    """Initialize and return the speaker diarization pipeline."""
    print("Loading speaker diarization model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv('HUGGINGFACE_ACCESS_TOKEN')
    )
    
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("Using GPU for diarization")
    
    return pipeline

def load_audio_file(audio_path):
    """Load audio file using pydub and return AudioSegment."""
    print("Loading audio for segmentation...")
    if audio_path.lower().endswith('.m4a'):
        return AudioSegment.from_file(audio_path, format="m4a")
    return AudioSegment.from_file(audio_path)

def perform_diarization(pipeline, audio_path):
    """Perform speaker diarization and return segments."""
    print("Performing speaker diarization...")
    diarization = pipeline(audio_path)
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })
    
    return segments

def filter_short_segments(segments, min_duration=1.0):
    """Filter out segments shorter than min_duration seconds."""
    filtered_segments = []
    for segment in segments:
        duration = segment["end"] - segment["start"]
        if duration >= min_duration:
            filtered_segments.append(segment)
        else:
            print(f"Ignoring short segment: Speaker {segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s)")
    return filtered_segments

def merge_segments(segments):
    """
    Merge all consecutive segments from the same speaker.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    # Sort segments by start time
    segments = sorted(segments, key=lambda x: x["start"])
    
    merged_segments = [segments[0].copy()]
    
    for current_segment in segments[1:]:
        last_segment = merged_segments[-1]
        
        if current_segment["speaker"] == last_segment["speaker"]:
            # Always merge if same speaker
            last_segment["end"] = max(last_segment["end"], current_segment["end"])
        else:
            # Different speaker - add as new segment
            merged_segments.append(current_segment.copy())
    
    print(f"Merged {len(segments)} segments into {len(merged_segments)} speaker turns")
    return merged_segments

def transcribe_segment(audio_segment, whisper_model):
    """Transcribe an audio segment using Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    audio_segment.export(temp_wav_path, format="wav")
    
    try:
        result = whisper_model.transcribe(temp_wav_path)
        text = result["text"]
    except Exception as e:
        text = f"[Transcription error: {e}]"
    finally:
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
    
    return text

def process_segments(audio, segments, whisper_model):
    """Process all segments and return transcription results."""
    results = []
    print("\nProcessing speaker segments...")
    
    for segment in segments:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        audio_segment = audio[start_ms:end_ms]
        
        text = transcribe_segment(audio_segment, whisper_model)
        
        results.append({
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "transcript": text
        })
        
        duration = segment["end"] - segment["start"]
        print(f"Speaker {segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s, duration: {duration:.1f}s): {text}")
    
    return results

def save_results(results, output_file):
    """Save transcription results to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Speaker Diarization and Transcription Results\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"Speaker {r['speaker']} ({r['start']:.1f}s - {r['end']:.1f}s):\n")
            f.write(f"{r['transcript']}\n\n")
    print(f"\nResults saved to: {output_file}")

def diarize_and_transcribe(audio_path, output_file=None):
    """
    Perform speaker diarization and transcription on an audio file.
    
    Args:
        audio_path: Path to the audio file
        output_file: Optional path to save the transcription results
    
    Returns:
        List of dictionaries containing speaker, start, end, and transcript
    """
    print(f"Processing audio file: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} not found")
        return None
    
    # Initialize models
    pipeline = initialize_diarization_pipeline()
    # print(whisper.available_models())
    whisper_model = whisper.load_model("small.en") # ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo']
    
    # Process audio
    audio = load_audio_file(audio_path)
    segments = perform_diarization(pipeline, audio_path)
    filtered_segments = filter_short_segments(segments)
    merged_segments = merge_segments(filtered_segments)
    results = process_segments(audio, merged_segments, whisper_model)
    
    # Save results if needed
    if output_file:
        save_results(results, output_file)
    
    return results

def format_transcript(results):
    """Format the transcription results for display."""
    if not results:
        return "No results to display"
    
    formatted = "Speaker Diarization and Transcription Results\n"
    formatted += "=" * 50 + "\n\n"
    
    for r in results:
        formatted += f"Speaker {r['speaker']} ({r['start']:.1f}s - {r['end']:.1f}s):\n"
        formatted += f"{r['transcript']}\n\n"
    
    return formatted

if __name__ == "__main__":
    audio_file = "recordings/1min.wav"
    output_file = os.path.join("text", "transcription_output.txt")
    
    start = time.time()
    results = diarize_and_transcribe(audio_file, output_file)
    end = time.time()
    
    if results:
        print("\n" + "=" * 50)
        print(f"Transcription complete ({end-start:.2f} seconds)")
        print(f"Total segments: {len(results)}")
        unique_speakers = len(set(r['speaker'] for r in results))
        print(f"Number of speakers: {unique_speakers}")
