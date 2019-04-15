import queue
# from collections import deque

from server.config.config import BUFFER_SIZE

class IPredictor:
        
    def refreshBuffer(self):
        self.buffer = queue.Queue(maxsize=BUFFER_SIZE)
        
    def predict(self):
        """
        Executes a particular predictor model

        This function is the wrapper called by the prediction thread in the audio tagger model.

        Parameters
        ----------

        Returns
        -------

        Notes
        ------
        At the end of the prediction,
        self.model.onNewPredictionCalculated(probabilityArray) is essential to inform the model
        about new prediction.

        """
        raise NotImplementedError
