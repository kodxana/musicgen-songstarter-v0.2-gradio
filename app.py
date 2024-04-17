import os
import gradio as gr
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path
import tempfile
import torchaudio

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Load the pretrained model
model = MusicGen.get_pretrained('nateraw/musicgen-songstarter-v0.2')

def generate_music(description, duration, melody_audio_path):
    model.set_generation_params(duration=duration)  # Set duration based on user input
    
    if melody_audio_path is not None:
        melody, sr = torchaudio.load(melody_audio_path)
        wav = model.generate_with_chroma([description], melody[None], sr)
    else:
        descriptions = [description] * 3  # Use the same description three times to generate 3 samples
        wav = model.generate(descriptions)  # Generate the waveform
    
    # Save the audio using the audio_write function from audiocraft
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        audio_path = f.name  # Generate a path for the temporary file
        # Write the audio file with normalization and loudness compression
        saved_path = audio_write(Path(audio_path), wav[0].cpu(), model.sample_rate,
                                 strategy="loudness", loudness_compressor=True)
        # Convert the path object to a string to ensure compatibility with Gradio
        saved_path_str = str(saved_path)
    
    return saved_path_str, saved_path_str

# Define Gradio interface
iface = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Enter your music description", placeholder="e.g., acoustic, guitar, melody, trap, d minor, 90 bpm"),
        gr.Slider(minimum=5, maximum=30, value=10, step=1, label="Duration (seconds)"),
        gr.Audio(label="Melody Audio (optional)", type="filepath")
    ],
    outputs=[
        gr.File(label="Download Generated Music"),
        gr.Audio(label="Listen to Generated Music", type="filepath")
    ],
    title="MusicGen Songstarter v0.2 Demo",
    description="Generate song ideas using the musicgen-songstarter-v0.2 model by nateraw. Gradio demo created by Madiator2011.",
    article="""
    <div>
        <p>This demo uses the <code>musicgen-songstarter-v0.2</code> model developed by <a href="https://huggingface.co/nateraw" target="_blank">nateraw</a>. The model is a fine-tuned version of <code>musicgen-stereo-melody-large</code> trained on a dataset of melody loops from nateraw's Splice sample library.</p>
        <p>If you find this model interesting, please consider:</p>
        <ul>
            <li>Following nateraw on <a href="https://github.com/nateraw" target="_blank">GitHub</a></li>
            <li>Following nateraw on <a href="https://twitter.com/nateraw" target="_blank">Twitter</a></li>
        </ul>
        <p>Gradio demo created by <a href="https://huggingface.co/Madiator2011" target="_blank">Madiator2011</a>.</p>
        <p>If you'd like to run this demo on RunPod and support Madiator2011, you can use the following referral link: <a href="https://runpod.io?ref=vfker49t" target="_blank">https://runpod.io?ref=vfker49t</a></p>
    </div>
    """,
    examples=[
        ["acoustic, guitar, melody, trap, d minor, 90 bpm", 10, None],
        ["ethereal, ambient, piano, 120 bpm", 20, None]
    ],
    allow_flagging="never"
)

# Launch the application
iface.launch(server_name="0.0.0.0", server_port=7860)
