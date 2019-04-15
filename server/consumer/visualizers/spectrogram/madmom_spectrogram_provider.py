import time
import numpy as np

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor, Processor

from server.consumer.visualizers.i_visualizer import IVisualizer


class MadmomSpectrogramProvider(IVisualizer):

    sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
    fsig_proc = FramedSignalProcessor(frame_size=1024, hop_size=128, origin='future')
    spec_proc = SpectrogramProcessor(frame_size=1024)
    filt_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
    processorPipeline = SequentialProcessor([sig_proc, fsig_proc, spec_proc, filt_proc])

    def __init__(self):
        super(MadmomSpectrogramProvider, self).__init__()
        self.sliding_window = np.zeros((128, 256), dtype=np.float32)

    def computeSpectrogram(self):
        if not self.buffer.empty():
        # if len(self.buffer) > 0:
            # frame = self.buffer.popleft()
            frame = self.buffer.get()
            
            spectrogram = self.processorPipeline.process(frame)
            # check if there is audio content
            frame = spectrogram[0]
            if np.any(np.isnan(frame)):
                frame = np.zeros_like(frame, dtype=np.float32)

            # update sliding window
            self.sliding_window[:, 0:-1] = self.sliding_window[:, 1::]
            self.sliding_window[:, -1] = frame
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
