from kokoro import KPipeline
import soundfile as sf
import numpy as np

# Inicializacija pipeline
pipeline = KPipeline(lang_code='a')

# Besedilo
text = '''
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''

# Uporabi glas af_heart_5
generator = pipeline(text, voice='af_heart')

# Združi vse avdio dele
audio_chunks = []
for _, _, audio in generator:
    audio_chunks.append(audio)

# Združi v en signal
full_audio = np.concatenate(audio_chunks)

# Shrani v datoteko
sf.write("kokoro_output.wav", full_audio, 24000)

print("✅ Audio saved to kokoro_output.wav")