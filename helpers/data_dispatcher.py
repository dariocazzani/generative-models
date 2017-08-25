#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Authors:    Dario Cazzani
"""

import numpy as np
import random
from helpers.signal_processing import floats_to_wav
import sys

def CMajorScaleDistribution(batch_size):
    sample_rate = 16000
    seconds = 1
    t = np.linspace(0, seconds, sample_rate*seconds)  # 16000 Hz sampling rate
    C_major_scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
    num_notes = 4
    note_length = int(sample_rate * seconds / num_notes)
    while True:
        try:
            batch = []
            for i in range(batch_size):
                # select random note
                sounds = []
                for note in range(num_notes):
                    pitch = C_major_scale[np.random.randint(len(C_major_scale))]
                    sound = np.sin(2*np.pi*t[:note_length]*pitch)
                    sounds.append(sound)
                sounds = np.concatenate(sounds).ravel()
                noise = [random.gauss(0.0, 1.0) for i in range(sample_rate*seconds)]
                noisy_sound = sounds + 0.08 * np.asarray(noise)
                batch.append(noisy_sound)
            for i in range(np.minimum(5, batch_size)):
                floats_to_wav('real_sample_{}.wav'.format(i), batch[i], 16000)
            yield np.asarray(batch)

        except Exception as e:
            print('Could not produce batch of sinusoids because: {}'.format(e))
            sys.exit()

def main():
    data = CMajorScaleDistribution(32)
    batch = data.__next__()
    # batch = data.next()
    print(batch.shape)

if __name__ == '__main__':
    main()
