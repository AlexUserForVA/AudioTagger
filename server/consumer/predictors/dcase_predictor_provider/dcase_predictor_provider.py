import os
import time
import torch
import torch.nn as nn
import numpy as np

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor

from server.config.config import PROJECT_ROOT, BUFFER_SIZE
from server.consumer.predictors.i_predictor import IPredictor
from server.consumer.predictors.dcase_predictor_provider.baseline_net import Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class DcasePredictorProvider(IPredictor):

    sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
    fsig_proc = FramedSignalProcessor(frame_size=1024, hop_size=128, origin='future')
    spec_proc = SpectrogramProcessor(frame_size=1024)
    filt_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
    processorPipeline = SequentialProcessor([sig_proc, fsig_proc, spec_proc, filt_proc])

    classes = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime",
               "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close",
               "Electric_piano",
               "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire",
               "Harmonica",
               "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone",
               "Scissors",
               "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet",
               "Violin_or_fiddle",
               "Writing"]

    def __init__(self):
        self.prediction_model = Net()
        self.prediction_model.load_state_dict(
            torch.load(os.path.join(PROJECT_ROOT, 'server/consumer/predictors/dcase_predictor_provider/baseline_net.pt'),  map_location=lambda storage, location: storage))
        self.prediction_model.to(device)
        self.prediction_model.eval()

        self.window = np.zeros((128, 256), dtype=np.float32)

        # start sliding window thread calculation
        # self.slidingWindowThread = SlidingWindowThread()

    def registerModel(self, model):
        self.model = model

    def predict(self, tGroundTruth):
        # print("Prediction started: " + str(time.time()))
        '''
        for i in range(256):
            ringBufferIndex = (tGroundTruth - i - 1) % BUFFER_SIZE
            frame = self.model.sharedMemory[ringBufferIndex]

            frame = np.fromstring(frame, np.int16)
            spectrogram = self.processorPipeline.process(frame)

            # check if there is audio content
            frame = spectrogram[0]
            if np.any(np.isnan(frame)):
                frame = np.zeros_like(frame, dtype=np.float32)

            # update sliding window
            self.window[:, 0:-1] = self.window[:, 1::]
            self.window[:, -1] = frame
        '''
        self.window = self.model.specProvider.sliding_window.copy()
        # print("Spectrogram calculated: " +  str(time.time()))
        input = self.window[np.newaxis, np.newaxis]
        cuda_torch_input = torch.from_numpy(input).to(device)
        model_output = self.prediction_model(cuda_torch_input)
        softmax = nn.Softmax(dim=1)
        softmax_output = softmax(model_output)
        predicts = softmax_output.cpu().detach().numpy().flatten()
        probs = [[elem, predicts[index].item(), index] for index, elem in enumerate(self.classes)]
        # print("Prediction finished: " +  str(time.time()))
        return probs
