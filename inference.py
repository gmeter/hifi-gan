from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
import soundfile as sf

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]




def get_mel(wav):
    """Convert waveform to Mel-spectrogram."""
    # Ensure audio is at the correct sample rate
    if h.sampling_rate != 22050:
        print(f"Warning: Model expects {h.sampling_rate}Hz audio")
    
    # Normalize audio to [-1, 1]
    wav = wav / max(abs(wav.min()), abs(wav.max()))
    
    mel = mel_spectrogram(wav, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
    print(f"Initial mel shape: {mel.shape}")
    
    # Convert to tensor and reshape
    mel = torch.FloatTensor(mel)
    mel = mel.squeeze(1)  # Remove the unnecessary dimension
    mel = mel.unsqueeze(0)  # Add batch dimension
    
    print(f"Final mel shape: {mel.shape}")
    return mel

def inference(args, config_file):
    """Run inference for HiFi-GAN."""
    # Load the config from the config file
    with open(config_file) as f:
        config = json.load(f)
    global h
    h = AttrDict(config)
    
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(args.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate through all WAV files in the input directory
    wav_files = glob.glob(os.path.join(args.input_wavs_dir, '*.wav'))
    for wav_path in wav_files:
        print(f"Processing {wav_path}")
        wav, sr = sf.read(wav_path)
        
        # Resample if necessary
        if sr != h.sampling_rate:
            print(f"Resampling from {sr} to {h.sampling_rate}")
            from librosa import resample
            wav = resample(y=wav, orig_sr=sr, target_sr=h.sampling_rate)
        
        wav = torch.FloatTensor(wav).unsqueeze(0).to(device)

        # Generate Mel-spectrogram
        mel = get_mel(wav)

        # Pass the Mel-spectrogram through the generator
        with torch.no_grad():
            y_g_hat = generator(mel)

        # Save the generated audio
        y_g_hat = y_g_hat.squeeze().cpu().numpy()
        
        # Normalize output audio
        y_g_hat = y_g_hat / max(abs(y_g_hat.min()), abs(y_g_hat.max()))

        # Save the output WAV file
        output_path = os.path.join(args.output_dir, os.path.basename(wav_path))
        sf.write(output_path, y_g_hat, h.sampling_rate)
        print(f"Saved to {output_path}")


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')

    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        print('Using GPU.')
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        print('Using CPU.')
        device = torch.device('cpu')

    inference(a, config_file)


if __name__ == '__main__':
    main()