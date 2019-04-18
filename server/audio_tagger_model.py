import time
import pyaudio
import numpy as np
from collections import deque

from pydoc import locate
from threading import Thread, Event

from server.producer.pyaudio_producer import MicrophoneThread, FileThread

from server.config.config import BUFFER_SIZE, START_FILE

class SpectrogramThread(Thread):

    def __init__(self, model, name='SpectrogramThread'):
        self.t = 0
        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        while not self._stopevent.isSet():
            if len(self.model.sharedMemory) > self.t:
                # time.sleep(0.029)
                spec = self.model.specProvider.computeSpectrogram(self.t)
                self.t += 1
                self.t = self.t % BUFFER_SIZE
                if spec is not None:
                    self.model.onNewSpectrogramCalculated(spec)

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        Thread.join(self, timeout)

class PredictionThread(Thread):

    def __init__(self, model, name='PredictionThread'):
        self.t = 0
        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        while not self._stopevent.isSet():
            if len(self.model.sharedMemory) > self.t:
                probs = self.model.predProvider.predict(self.t)
                self.t += 1
                self.t = self.t % BUFFER_SIZE
                if probs is not None:
                    self.model.onNewPredictionCalculated(probs)

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        Thread.join(self, timeout)

'''
class AudioThread(Thread):

    def __init__(self, model, name='AudioThread'):
        self.t = 0
        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        P = pyaudio.PyAudio()
        stream = P.open(rate=32000, format=pyaudio.paInt16, channels=1, output=True)
        while not self._stopevent.isSet():
            if len(self.model.sharedMemory) > 0:
                for i in range(902):
                    time.sleep(0.032)
                    x = self.model.sharedMemory[i][1]
                    stream.write(self.model.sharedMemory[i][1].tobytes())
                stream.close()  # this blocks until sound finishes playing
                P.terminate()
                return

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        Thread.join(self, timeout)
'''

class AudioThread(Thread):

    def __init__(self, model, name='AudioThread'):
        self.t = 0
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                channels=1,
                rate=32000,
                output=True)

        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        while not self._stopevent.isSet():
            if len(self.model.sharedMemory) > self.t:
                chunk = self.model.sharedMemory[self.t][1]
                self.stream.write(chunk)
                self.t += 1
                self.t = self.t % BUFFER_SIZE

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

        self.t = 0
        self.sharedMemory = deque(maxlen=BUFFER_SIZE)

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
            self.producerThread = FileThread(self, filePath)

        self.audioThread = AudioThread(self)
        self.specThread = SpectrogramThread(self)
        # self.predThread = PredictionThread(self)
        self.producerThread.start()
        self.audioThread.start()
        self.specThread.start()
        # self.predThread.start()

    ############ Refresh function #############
    def refreshAudioTagger(self, settings):
        self.audioThread.join()
        self.specThread.join()
        # self.predThread.join()
        self.producerThread.join()

        self.t = 0

        # Restart audio tagger with delivered settings
        isLive = settings['isLive']
        file = settings['file']
        predictor = settings['predictor']

        if isLive:
            self.producerThread = MicrophoneThread(self)
        else:
            filePath = [elem['path'] for elem in self.getSourceList() if elem['id'] == file][0]
            self.producerThread = FileThread(self, filePath)
            self.audioThread = AudioThread(self)

        predictorClassPath = [elem['predictorClassPath'] for elem in self.getPredList() if elem['id'] == predictor][0]
        predProviderClass = locate('server.consumer.predictors.{}'.format(predictorClassPath))
        newPredProvider = predProviderClass()
        self.setPredProvider(newPredProvider)
        self.predProvider.registerModel(self)

        self.sharedMemory.clear()

        self.producerThread.start()
        self.specThread = SpectrogramThread(self)
        self.specThread.start()
        if not isLive:
            self.audioThread.start()

        # self.predThread = PredictionThread(self)
        # self.predThread.start()

    def putToSM(self, chunk):
        self.sharedMemory.append((self.t, chunk))
        self.t += 1
        self.t = self.t % BUFFER_SIZE
