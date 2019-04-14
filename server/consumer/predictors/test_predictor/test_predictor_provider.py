import random
from server.consumer.predictors.i_predictor import IPredictor

class TestPredictorProvider(IPredictor):

    classes = ["Football", "Basketball", "Ice Hockey"]

    def predict(self):
        # insert a real predictor
        probs = [[elem, random.uniform(0, 1), index] for index, elem in enumerate(self.classes)]
        return probs