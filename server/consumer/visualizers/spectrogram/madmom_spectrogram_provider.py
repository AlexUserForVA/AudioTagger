import time
import numpy as np

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor, Processor

from server.config.config import BUFFER_SIZE
from server.consumer.visualizers.i_visualizer import IVisualizer


class MadmomSpectrogramProvider(IVisualizer):

    sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
    fsig_proc = FramedSignalProcessor(frame_size=1024, hop_size=128, origin='future')
    spec_proc = SpectrogramProcessor(frame_size=1024)
    filt_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
    processorPipeline = SequentialProcessor([sig_proc, fsig_proc, spec_proc, filt_proc])

    def __init__(self):
        self.sliding_window = np.zeros((128, 256), dtype=np.float32)
        self.lastProceededGroundTruth = None

    def registerModel(self, model):
        self.model = model

    def computeSpectrogram(self, tGroundTruth):
        if tGroundTruth != self.lastProceededGroundTruth:
            frame = self.model.sharedMemory[(tGroundTruth - 1) % BUFFER_SIZE]
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


        return self.sliding_window.copy()
