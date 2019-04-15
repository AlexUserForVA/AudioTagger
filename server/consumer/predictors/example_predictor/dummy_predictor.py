import random
from server.consumer.predictors.i_predictor import IPredictor

class DummyPredictor(IPredictor):

    def predict(self):
        if not self.buffer.empty():
            _ = self.buffer.get()
            probs = [[elem, random.uniform(0, 1), index] for index, elem in enumerate(["Class 1", "Class 2", "Class 3"])]
            return probs