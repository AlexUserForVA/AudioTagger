import random
from server.consumer.predictors.i_predictor import IPredictor

class TestPredictorProvider(IPredictor):

    classes = ["Football", "Basketball", "Ice Hockey"]

    def registerModel(self, model):
        self.model = model

    def predict(self, t):
        probs = [[elem, random.uniform(0, 1), index] for index, elem in enumerate(self.classes)]
        return probs