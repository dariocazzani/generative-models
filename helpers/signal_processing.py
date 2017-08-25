import wave
import struct
from scipy.io.wavfile import write
import numpy as np

def floats_to_wav(filename, audio, fs):
    scaled = np.int16(audio/np.max(np.abs(audio)) * np.iinfo(np.int16).max)
    write(filename, fs, scaled)

def wav_to_floats(wave_file):
    w = wave.open(wave_file, 'rb')
    astr = w.readframes(w.getnframes())
    a = struct.unpack("%ih" % (len(astr) / 2), astr)
    a = [float(val) / pow(2, 15) for val in a]
    w.close()
    return np.asarray(a), w.getframerate()
