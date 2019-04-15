import random
from server.consumer.predictors.i_predictor import IPredictor

class TestPredictorProvider(IPredictor):

    classes = ["Football", "Basketball", "Ice Hockey"]

    def predict(self):
        # insert a real predictor
        if not self.buffer.empty():
            _ = self.buffer.get()
            probs = [[elem, random.uniform(0, 1), index] for index, elem in enumerate(self.classes)]
            return probs