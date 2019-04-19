import time
import numpy as np

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor, Processor

from server.consumer.visualizers.i_visualizer import IVisualizer


class MadmomSpectrogramProvider(IVisualizer):

    sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
    fsig_proc = FramedSignalProcessor(frame_size=512, hop_size=128, origin='future')
    spec_proc = SpectrogramProcessor(frame_size=512)
    filt_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
    processorPipeline = SequentialProcessor([sig_proc, fsig_proc, spec_proc, filt_proc])

    def __init__(self):
        self.sliding_window = np.zeros((103, 256), dtype=np.float32)
        self.lastProceededGroundTruth = None

    def registerModel(self, model):
        self.model = model

    '''
    cur_window = np.zeros((128, 256), dtype=np.float32)

    def computeSpectrogramFull(self):
        print(time.time())
        if len(self.model.sharedMemory) > 256 + self.t:
            for i in range(256):
                frame = self.processorPipeline.process(self.model.sharedMemory[self.t + i][1])[0]

            # check if there is audio content
            # frame = spectrogram[0]
            # if np.any(np.isnan(frame)):
            #    frame = np.zeros_like(frame, dtype=np.float32)

            # update sliding window
                self.cur_window[:, 0:-1] = self.cur_window[:, 1::]
                self.cur_window[:, -1] = frame

            # _ = self.buffer.get()
            self.t += 1
            return self.cur_window.copy()
    '''

    def computeSpectrogram(self, tGroundTruth):
        # print(time.time())
        if tGroundTruth != self.lastProceededGroundTruth:
            frame = self.model.sharedMemory[tGroundTruth - 1]
            frame = np.fromstring(frame, np.int16)
            spectrogram = self.processorPipeline.process(frame)
            # check if there is audio content
            frame = spectrogram[0]
            if np.any(np.isnan(frame)):
                frame = np.zeros_like(frame, dtype=np.float32)

            # update sliding window
            self.sliding_window[:, 0:-1] = self.sliding_window[:, 1::]
            self.sliding_window[:, -1] = frame

            self.lastProceededGroundTruth = tGroundTruth

            # self.counter += 1
            # print("Spectrogram: " + str(self.counter))

        return self.sliding_window.copy()
