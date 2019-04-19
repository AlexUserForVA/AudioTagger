import random

from threading import Thread, Event

from server.consumer.predictors.predictor_contract import PredictorContract

class PredictionThread(Thread):

    def __init__(self, provider, name='PredictionThread'):
        self.provider = provider
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        while not self._stopevent.isSet():
            if len(self.provider.model.sharedMemory) > 0:
                probs = self.provider.model.predProvider.predict()
                if probs is not None:
                    self.provider.model.onNewPredictionCalculated(probs)

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        Thread.join(self, timeout)

class DummyPredictor(PredictorContract):

    def registerModel(self, model):
        self.model = model

    def start(self):
        self.predThread = PredictionThread(self)
        self.predThread.start()

    def stop(self):
        self.predThread.join()

    def predict(self):
        probs = [[elem, random.uniform(0, 1), index] for index, elem in enumerate(["Class 1", "Class 2", "Class 3"])]
        return probs