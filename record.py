import pyaudio
import wave

def record_audio(filename='test.wav', duration=3):
    """
    Records audio from the microphone and saves it to a .wav file.

    Args:
        filename (str): Output filename (default: 'test.wav')
        duration (int): Duration of recording in seconds (default: 3)
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()

    print("[INFO] Recording started...")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * duration))]

    print("[INFO] Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
