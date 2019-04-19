import numpy as np

from threading import Thread, Event

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor

from server.config.config import BUFFER_SIZE
from server.consumer.visualizers.visualisation_contract import VisualisationContract


class VisualisationThread(Thread):

    def __init__(self, provider, name='VisualisationThread'):
        self.provider = provider
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        while not self._stopevent.isSet():
            if len(self.provider.model.sharedMemory) > 0:
                spec = self.provider.computeSpectrogram(self.provider.model.tGroundTruth)
                self.provider.model.onNewSpectrogramCalculated(spec)

    def join(self, timeout=None):
        self._stopevent.set()
        Thread.join(self, timeout)


class MadmomSpectrogramProvider(VisualisationContract):

    sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
    fsig_proc = FramedSignalProcessor(frame_size=1024, hop_size=128, origin='future')
    spec_proc = SpectrogramProcessor(frame_size=1024)
    filt_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
    processorPipeline = SequentialProcessor([sig_proc, fsig_proc, spec_proc, filt_proc])

    def __init__(self):
        self.sliding_window = np.zeros((128, 256), dtype=np.float32)
        self.lastProceededGroundTruth = None

    def start(self):
        self.visThread = VisualisationThread(self)
        self.visThread.start()

    def stop(self):
        self.visThread.join()

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
