import time
import numpy as np

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor, Processor

from server.consumer.visualizers.i_visualizer import IVisualizer
from server.config.config import BUFFER_SIZE


class MadmomSpectrogramProvider(IVisualizer):

    t = 0

    sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
    fsig_proc = FramedSignalProcessor(frame_size=1024, hop_size=128, origin='future')
    spec_proc = SpectrogramProcessor(frame_size=1024)
    filt_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
    processorPipeline = SequentialProcessor([sig_proc, fsig_proc, spec_proc, filt_proc])

    def __init__(self):
        self.sliding_window = np.zeros((128, 256), dtype=np.float32)

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

    def computeSpectrogram(self, t):
        # print(time.time())
        # t = self.model.tGroundTruth
        frame = self.model.sharedMemory[t]
        frame = np.fromstring(frame, np.int16)
        spectrogram = self.processorPipeline.process(frame)
        # check if there is audio content
        frame = spectrogram[0]
        if np.any(np.isnan(frame)):
            frame = np.zeros_like(frame, dtype=np.float32)

        # update sliding window
        self.sliding_window[:, 0:-1] = self.sliding_window[:, 1::]
        self.sliding_window[:, -1] = frame

        # self.counter += 1
        # print("Spectrogram: " + str(self.counter))

        return self.sliding_window.copy()
    '''
    # function stems from madmom (https://github.com/CPJKU/madmom/blob/master/madmom/processors.py)
    def process_online(self, processor, infile, outfile, **kwargs):

        # set arguments for online processing
        # Note: pass only certain arguments, because these will be passed to the
        #       processors at every time step (kwargs contains file handles etc.)
        process_args = {'reset': False}  # do not reset stateful processors
        # process everything frame-by-frame
        for frame in stream:
            if self.stopFlag:
                break
            _process((processor, frame, outfile, process_args))

    def run(self):
        processor = IOProcessor(in_processor=processor_pipeline2, out_processor=self.output_processor)
        # process_online(processor, infile='/home/alex/pycharm_projects/AEC_AudioTaggerGUI/server/files/trumpet.wav', outfile=None, sample_rate=32000)
        # self.process_online(processor, infile=audioSource, outfile=None, sample_rate=32000)
    

    def output_processor(self, data, output):
        """
        Output data processor
        """
        # check if there is audio content
        frame = data[0]
        if np.any(np.isnan(frame)):
            frame = np.zeros_like(frame, dtype=np.float32)

        # increase frame count
        self.frame_count += 1
        self.frame_count = np.mod(self.frame_count, self.predict_every_k)
        do_invoke = self.frame_count == 0

        # update sliding window
        self.sliding_window[:, 0:-1] = self.sliding_window[:, 1::]
        self.sliding_window[:, -1] = frame

        # invoke model that new spectrogram is available
        self.model.onNewSpectrogramCalculated(self.sliding_window.copy())
    '''
