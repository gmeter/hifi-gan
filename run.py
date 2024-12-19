import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import sys
import json
from models import Generator  # Correct import statement

class AttrDict(dict):
    """A dictionary that provides attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class AudioGenderConverter:
    def __init__(self, model_path, config_path):
        # Load the HiFi-GAN configuration from a JSON file
        with open(config_path) as f:
            config = json.load(f)
        h = AttrDict(config)
        
        # Load the HiFi-GAN model
        self.model = Generator(h).to('cpu')
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')['generator'])
        self.model.eval()
        self.model.remove_weight_norm()

    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        audio, sr = librosa.load(audio_path, sr=16000)
        return audio, sr

    def parse_textgrid(self, textgrid_path):
        """Parse the TextGrid file to get intervals."""
        intervals = []
        with open(textgrid_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "xmin" in line:
                    xmin = float(line.split('=')[1].strip())
                if "xmax" in line:
                    xmax = float(line.split('=')[1].strip())
                if "text" in line:
                    text = line.split('=')[1].strip().strip('"')
                    intervals.append((xmin, xmax, text))
        return intervals

    def rms_normalize(self, audio_segment, target_dBFS=-20.0):
        """Normalize audio segment to the target dBFS."""
        rms = np.sqrt(np.mean(audio_segment**2))
        target_rms = 10**(target_dBFS / 20.0)
        scaling_factor = target_rms / (rms + 1e-6)  # Add a small value to avoid division by zero
        normalized_audio = audio_segment * scaling_factor
        return normalized_audio

    def apply_limiter(self, audio_segment, threshold=0.9):
        """Apply a limiter to the audio segment to prevent clipping."""
        return np.clip(audio_segment, -threshold, threshold)

    def modify_voice_characteristics(self, audio_segment):
        """Modify voice characteristics using HiFi-GAN."""
        # Normalize the audio segment before processing
        audio_segment = self.rms_normalize(audio_segment)
        print(f"Before limiting - mean: {np.mean(audio_segment)}, max: {np.max(audio_segment)}, min: {np.min(audio_segment)}")
        
        # Apply limiter to prevent extreme values
        audio_segment = self.apply_limiter(audio_segment)
        print(f"After limiting - mean: {np.mean(audio_segment)}, max: {np.max(audio_segment)}, min: {np.min(audio_segment)}")
        
        # Convert the audio segment to a mel spectrogram
        mel_spectrogram = self.audio_to_mel(audio_segment)
        
        # Apply the HiFi-GAN vocoder
        with torch.no_grad():
            generated_audio = self.model(mel_spectrogram).cpu().numpy()
        
        # Reshape the generated audio to a 1-dimensional array
        generated_audio = generated_audio.reshape(-1)
        print(f"After processing - mean: {np.mean(generated_audio)}, max: {np.max(generated_audio)}, min: {np.min(generated_audio)}")
        
        # Scale down the output to prevent clipping
        generated_audio = np.clip(generated_audio, -0.9, 0.9)
        
        # Normalize the generated audio before returning
        generated_audio = self.rms_normalize(generated_audio)
        print(f"After normalization - mean: {np.mean(generated_audio)}, max: {np.max(generated_audio)}, min: {np.min(generated_audio)}")
        
        return generated_audio

    def audio_to_mel(self, audio_segment):
        """Convert audio segment to mel spectrogram."""
        # Ensure the audio segment is long enough for the FFT parameters
        min_length = 1024  # Minimum length for FFT
        if len(audio_segment) < min_length:
            pad_length = min_length - len(audio_segment)
            audio_segment = np.pad(audio_segment, (0, pad_length), 'constant')

        # Initialize mel spectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        # Convert audio to tensor and apply mel transform
        audio_tensor = torch.tensor(audio_segment).unsqueeze(0)
        mel_spectrogram = mel_transform(audio_tensor)
        return mel_spectrogram

    def process_audio(self, input_audio_path, textgrid_path, output_path):
        """Main processing pipeline."""
        # Load audio
        audio, sr = self.load_audio(input_audio_path)
        print(f"Original audio - mean: {np.mean(audio)}, max: {np.max(audio)}, min: {np.min(audio)}")
        
        # Parse TextGrid file
        intervals = self.parse_textgrid(textgrid_path)
        
        # Process audio in chunks based on word alignment
        modified_audio_segments = []
        last_end_sample = 0
        
        for xmin, xmax, text in intervals:
            start_sample = int(round(xmin * sr))
            end_sample = int(round(xmax * sr))
            
            # Handle gap between the last end sample and the current start sample
            if start_sample > last_end_sample:
                gap_segment = audio[last_end_sample:start_sample]
                modified_audio_segments.append(gap_segment)
            
            audio_segment = audio[start_sample:end_sample]
            
            if text:  # Only modify intervals with text
                modified_segment = self.modify_voice_characteristics(audio_segment)
                original_length = end_sample - start_sample
                
                # Ensure exact length match by interpolation if needed
                if len(modified_segment) != original_length:
                    if len(modified_segment) > 0:
                        stretch_factor = len(modified_segment) / original_length
                        modified_segment = librosa.effects.time_stretch(modified_segment, rate=stretch_factor)
                    
                    # Trim or pad the modified segment to match the original length
                    if len(modified_segment) > original_length:
                        modified_segment = modified_segment[:original_length]
                    elif len(modified_segment) < original_length:
                        pad_length = original_length - len(modified_segment)
                        
                        # Ensure modified_segment is 1-dimensional before padding
                        if modified_segment.ndim == 1:
                            modified_segment = np.pad(modified_segment, (0, pad_length), 'constant')
                        else:
                            print(f"Warning: modified_segment is not 1-dimensional, shape: {modified_segment.shape}. Skipping padding.")
            
            else:
                modified_segment = audio_segment  # Set it to the original segment

            print(f"Segment after modification - mean: {np.mean(modified_segment)}, max: {np.max(modified_segment)}, min: {np.min(modified_segment)}")
            modified_audio_segments.append(modified_segment)
            
            last_end_sample = end_sample
        
        # Handle any remaining audio after the last interval
        if last_end_sample < len(audio):
            remaining_segment = audio[last_end_sample:]
            modified_audio_segments.append(remaining_segment)
        
        # Concatenate all modified audio segments
        modified_audio = np.concatenate(modified_audio_segments)
        print(f"After concatenation - mean: {np.mean(modified_audio)}, max: {np.max(modified_audio)}, min: {np.min(modified_audio)}")
        
        # Apply final processing to remove any leading or trailing silence
        modified_audio, _ = librosa.effects.trim(modified_audio)
        print(f"After trimming - mean: {np.mean(modified_audio)}, max: {np.max(modified_audio)}, min: {np.min(modified_audio)}")

        # Normalize the entire audio to avoid excessive loudness
        modified_audio = self.rms_normalize(modified_audio)
        print(f"Final normalized audio - mean: {np.mean(modified_audio)}, max: {np.max(modified_audio)}, min: {np.min(modified_audio)}")
        
        # Save result
        sf.write(output_path, modified_audio, sr)
        
        return output_path

def main():
    # Example usage
    if len(sys.argv) != 6:
        print(f"{len(sys.argv)} Usage: python run.py <input_audio_file> <textGrid file> <output file> <model_path> <config_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    transcript = sys.argv[2]
    output_path = sys.argv[3]
    model_path = sys.argv[4]
    config_path = sys.argv[5]

    converter = AudioGenderConverter(model_path, config_path)
    converted_file = converter.process_audio(input_path, transcript, output_path)
    print(f"Converted audio saved to: {converted_file}")

if __name__ == "__main__":
    main()