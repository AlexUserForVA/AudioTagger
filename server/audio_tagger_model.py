import time
import wave
import pyaudio
import numpy as np
from collections import deque

from pydoc import locate
from threading import Thread, Event

# from server.producer.pyaudio_producer import MicrophoneThread, FileThread

from server.config.config import BUFFER_SIZE, START_FILE

class MicrophoneThread(Thread):

    CHUNK_SIZE = 1024
    SAMPLE_RATE = 32000

    def __init__(self, model, name='MicrophoneThread'):
        self.p = pyaudio.PyAudio()
        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=self.SAMPLE_RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK_SIZE)

        while not self._stopevent.isSet():
            chunk = stream.read(self.CHUNK_SIZE)
            self.model.putToSM(chunk)
            self.model.tGroundTruth += 1
            self.model.tGroundTruth = self.model.tGroundTruth % BUFFER_SIZE


    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        Thread.join(self, timeout)


class VisualisationThread(Thread):

    def __init__(self, model, name='VisualisationThread'):
        self.model = model
        self.t = 0
        self._stopevent = Event()
        Thread.__init__(self, name=name)
    '''
    def run(self):
        while not self._stopevent.isSet():
            if len(self.model.sharedMemory) > self.t:
                # print("Spectime: " + str(time.time()) + "GroundTruth: " + str(self.model.tGroundTruth))
                spec = self.model.specProvider.computeSpectrogram(self.t)
                self.t += 1
                self.t = self.t % BUFFER_SIZE
                if spec is not None:
                    self.model.onNewSpectrogramCalculated(spec)
    '''
    def run(self):
        while not self._stopevent.isSet():
            if len(self.model.sharedMemory) > 0:
                # print("Spectime: " + str(time.time()) + "GroundTruth: " + str(self.model.tGroundTruth))
                spec = self.model.specProvider.computeSpectrogram(self.model.tGroundTruth)
                if spec is not None:
                    self.model.onNewSpectrogramCalculated(spec)

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        Thread.join(self, timeout)

class PredictionThread(Thread):

    def __init__(self, model, name='PredictionThread'):
        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        while not self._stopevent.isSet():
            if len(self.model.sharedMemory) > 256:
                probs = self.model.predProvider.predict(self.model.tGroundTruth)
                if probs is not None:
                    self.model.onNewPredictionCalculated(probs)

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        Thread.join(self, timeout)

class AudioThread(Thread):

    CHUNK_SIZE = 1024

    def __init__(self, model, filePath, name='AudioThread'):
        self.p = pyaudio.PyAudio()
        self.wf = wave.open(filePath, 'rb')
        self.stream = self.p.open(format=pyaudio.paInt16,
                channels=1,
                rate=32000,
                output=True)

        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        chunk = self.wf.readframes(self.CHUNK_SIZE)
        while not self._stopevent.isSet() and chunk != b'':
            self.stream.write(chunk)
            self.model.putToSM(chunk)
            self.model.tGroundTruth += 1
            self.model.tGroundTruth = self.model.tGroundTruth % BUFFER_SIZE
            chunk = self.wf.readframes(self.CHUNK_SIZE)

        self.stream.stop_stream()
        self.stream.close()

        self.p.terminate()

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        Thread.join(self, timeout)

class AudioTaggerModel:

    def __init__(self, specProvider, predProvider, predList, sourceList):
        self.specProvider = specProvider
        self.predProvider = predProvider

        # initialization
        self.liveSpec = np.zeros((128, 256), dtype=np.float32)
        self.livePred = [["Class{}".format(index), 0.2, index] for index in range(10)]

        self.specProvider.registerModel(self)
        self.predProvider.registerModel(self)

        self.predList = predList
        self.sourceList = sourceList

        self.sharedMemory = deque(maxlen=BUFFER_SIZE)

        self.tGroundTruth = 0

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
        if START_FILE == None:
            self.producerThread = MicrophoneThread(self)
        else:
            filePath = [elem['path'] for elem in self.getSourceList() if elem['id'] == START_FILE][0]
            self.producerThread = AudioThread(self, filePath)

        self.specThread = VisualisationThread(self)
        self.predThread = PredictionThread(self)
        self.producerThread.start()
        self.specThread.start()
        self.predThread.start()

    ############ Refresh function #############
    def refreshAudioTagger(self, settings):
        self.specThread.join()
        self.predThread.join()
        self.producerThread.join()

        self.tGroundTruth = 0

        # Restart audio tagger with delivered settings
        isLive = settings['isLive']
        file = settings['file']
        predictor = settings['predictor']

        if isLive:
            self.producerThread = MicrophoneThread(self)
        else:
            filePath = [elem['path'] for elem in self.getSourceList() if elem['id'] == file][0]
            self.producerThread = AudioThread(self, filePath)

        predictorClassPath = [elem['predictorClassPath'] for elem in self.getPredList() if elem['id'] == predictor][0]
        predProviderClass = locate('server.consumer.predictors.{}'.format(predictorClassPath))
        newPredProvider = predProviderClass()
        self.setPredProvider(newPredProvider)
        self.predProvider.registerModel(self)

        self.sharedMemory.clear()

        self.producerThread.start()
        self.specThread = VisualisationThread(self)
        self.specThread.start()

        self.predThread = PredictionThread(self)
        self.predThread.start()

    def putToSM(self, chunk):
        self.sharedMemory.append(chunk)
