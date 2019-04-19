class PredictorContract:
        
    def registerModel(self, model):
        self.model = model

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

