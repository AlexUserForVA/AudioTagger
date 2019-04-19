import wave
import pyaudio
import numpy as np
from collections import deque

from pydoc import locate
from threading import Thread, Event

from server.config.config import BUFFER_SIZE, START_FILE, CHUNK_SIZE, SAMPLE_RATE, N_CHANNELS


class MicrophoneThread(Thread):

    def __init__(self, model, name='MicrophoneThread'):
        self.p = pyaudio.PyAudio()
        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=N_CHANNELS,
                             rate=SAMPLE_RATE,
                             input=True,
                             frames_per_buffer=CHUNK_SIZE)

        while not self._stopevent.isSet():
            chunk = stream.read(CHUNK_SIZE)
            self.model.putToSM(chunk)   # insert new chunk into shared memory
            self.model.tGroundTruth += 1    # increment global timestamp variable
            self.model.tGroundTruth = self.model.tGroundTruth % BUFFER_SIZE # ring buffer implementation

    def join(self, timeout=None):
        self._stopevent.set()
        Thread.join(self, timeout)


class AudiofileThread(Thread):

    def __init__(self, model, filePath, name='AudiofileThread'):
        self.p = pyaudio.PyAudio()
        self.wf = wave.open(filePath, 'rb')
        self.stream = self.p.open(format=pyaudio.paInt16,
                channels=N_CHANNELS,
                rate=SAMPLE_RATE,
                output=True)

        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        chunk = self.wf.readframes(CHUNK_SIZE)
        while not self._stopevent.isSet() and chunk != b'':
            self.stream.write(chunk)
            self.model.putToSM(chunk)   # insert new chunk into shared memory
            self.model.tGroundTruth += 1    # increment global timestamp variable
            self.model.tGroundTruth = self.model.tGroundTruth % BUFFER_SIZE # ring buffer implementation
            chunk = self.wf.readframes(CHUNK_SIZE)

        self.stream.stop_stream()
        self.stream.close()

        self.p.terminate()

    def join(self, timeout=None):
        self._stopevent.set()
        Thread.join(self, timeout)


class AudioTaggerManager:

    def __init__(self, specProvider, predProvider, predList, sourceList):
        self.visProvider = specProvider
        self.predProvider = predProvider

        # initialization of visualization and prediction output
        self.liveSpec = np.zeros((128, 256), dtype=np.float32)
        self.livePred = [["Class{}".format(index), 0.2, index] for index in range(10)]

        # consumers inform manager if an audio chunk is processed
        self.visProvider.registerModel(self)
        self.predProvider.registerModel(self)

        # selectable audio files and predictors
        self.predList = predList
        self.sourceList = sourceList

        self.sharedMemory = deque(maxlen=BUFFER_SIZE)

        self.tGroundTruth = 0   # global timestamp to keep up synchronization of consumers

        self.startThreads()

    def getPredList(self):
        return self.predList

    def getSourceList(self):
        return self.sourceList

    def setPredProvider(self, predProvider):
        self.predProvider = predProvider

    def getLiveSpectrogram(self):
        return self.liveSpec

    def getLivePrediction(self):
        return self.livePred

    def onNewSpectrogramCalculated(self, image):
        self.liveSpec = image

    def onNewPredictionCalculated(self, prob_dict):
        self.livePred = prob_dict

    def startThreads(self):

        # decide if audio input comes from microphone or file
        if START_FILE == None:
            self.producerThread = MicrophoneThread(self)
        else:
            filePath = [elem['path'] for elem in self.getSourceList() if elem['id'] == START_FILE][0]
            self.producerThread = AudiofileThread(self, filePath)

        self.producerThread.start()

        # start consumers
        self.visProvider.start()
        self.predProvider.start()

    ############ Refresh function #############
    # This function is called when the frontend
    # changes audio mode, file or predictor
    def refreshAudioTagger(self, settings):
        # stop producer and consumers
        self.producerThread.join()

        self.visProvider.stop()
        self.predProvider.stop()

        # reset ring buffer
        self.tGroundTruth = 0
        self.sharedMemory.clear()

        # restart audio tagger with delivered settings
        isLive = settings['isLive']
        file = settings['file']
        predictor = settings['predictor']

        # decide if audio input comes from microphone or file
        if isLive:
            self.producerThread = MicrophoneThread(self)
        else:
            filePath = [elem['path'] for elem in self.getSourceList() if elem['id'] == file][0]
            self.producerThread = AudiofileThread(self, filePath)

        # load selected prediction class via reflection, update references for the new predictor object
        predictorClassPath = [elem['predictorClassPath'] for elem in self.getPredList() if elem['id'] == predictor][0]
        predProviderClass = locate('server.consumer.predictors.{}'.format(predictorClassPath))
        newPredProvider = predProviderClass()
        self.setPredProvider(newPredProvider)
        self.predProvider.registerModel(self)

        # restart producer and consumers
        self.producerThread.start()

        self.visProvider.start()
        self.predProvider.start()

    def putToSM(self, chunk):
        if len(self.sharedMemory) < BUFFER_SIZE:
            self.sharedMemory.append(chunk)
        else:
            self.sharedMemory[self.tGroundTruth] = chunk
