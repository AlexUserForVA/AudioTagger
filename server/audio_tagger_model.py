import time
import numpy as np

from pydoc import locate
from threading import Thread, Event, Barrier

from server.config.config import N_PARALLEL_CONSUMERS
from server.producer.signal_provider import ProducerThread

class SpectrogramThread(Thread):

    def __init__(self, model, name='SpectrogramThread'):
        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        while not self._stopevent.isSet():
            try:
                time.sleep(128.0 / 32000.0) if self.model.liveMode else time.sleep(441.0 / 32000.0)
                spec = self.model.specProvider.computeSpectrogram()
                self.model.syncBarrier.wait()
                if spec is not None:
                    self.model.onNewSpectrogramCalculated(spec)
            except:
                pass

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        self.model.syncBarrier.abort()
        Thread.join(self, timeout)


class PredictionThread(Thread):

    def __init__(self, model, name='PredictionThread'):
        self.model = model
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        while not self._stopevent.isSet():
            try:
                probs = self.model.predProvider.predict()
                self.model.syncBarrier.wait()
                if probs is not None:
                    self.model.onNewPredictionCalculated(probs)
            except:
                pass

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        self.model.syncBarrier.abort()
        Thread.join(self, timeout)


class AudioTaggerModel:

    def __init__(self, producer, specProvider, predProvider, predList, sourceList):
        self.specProvider = specProvider
        self.predProvider = predProvider
        self.producer = producer

        # initialization
        self.liveSpec = np.zeros((128, 256), dtype=np.float32)
        self.livePred = [["Class{}".format(index), 0.2, index] for index in range(10)]

        producer.registerModel(self)

        self.predList = predList
        self.sourceList = sourceList

        self.liveMode = 1

        self.syncBarrier = Barrier(N_PARALLEL_CONSUMERS)

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
        self.producerThread = ProducerThread(self, None)
        self.specThread = SpectrogramThread(self)
        self.predThread = PredictionThread(self)
        self.producerThread.start()
        self.specThread.start()
        self.predThread.start()

    ############ Refresh function #############
    def refreshAudioTagger(self, settings):
        self.predThread.join()
        self.specThread.join()
        self.producerThread.join()

        self.syncBarrier = Barrier(N_PARALLEL_CONSUMERS)

        # Restart audio tagger with delivered settings
        isLive = settings['isLive']
        file = settings['file']
        predictor = settings['predictor']

        if isLive:
            filePath = None
            self.liveMode = 1
        else:
            self.liveMode = 0
            filePath = [elem['path'] for elem in self.getSourceList() if elem['id'] == file][0]

        predictorClassPath = [elem['predictorClassPath'] for elem in self.getPredList() if elem['id'] == predictor][0]
        predProviderClass = locate('server.consumer.predictors.{}'.format(predictorClassPath))
        newPredProvider = predProviderClass()
        self.setPredProvider(newPredProvider)

        # self.specProvider.buffer.clear()
        # self.predProvider.buffer.clear()
        self.specProvider.refreshBuffer()
        self.predProvider.refreshBuffer()

        # self.syncBarrier.reset()

        self.producerThread = ProducerThread(self, filePath)
        self.producerThread.start()
        self.specThread = SpectrogramThread(self, filePath)
        self.specThread.start()
        self.predThread = PredictionThread(self)
        self.predThread.start()

    def put_signal(self, signal):
        # self.specProvider.buffer.append(signal)
        # self.predProvider.buffer.append(signal)
        self.specProvider.buffer.put(signal)
        self.predProvider.buffer.put(signal)
