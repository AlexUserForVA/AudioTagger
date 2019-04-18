import random
from server.consumer.predictors.i_predictor import IPredictor

class DummyPredictor(IPredictor):

    def registerModel(self, model):
        self.model = model

    def predict(self, t):
        probs = [[elem, random.uniform(0, 1), index] for index, elem in enumerate(["Class 1", "Class 2", "Class 3"])]
        return probs