import Queue
# from collections import deque

from server.config.config import BUFFER_SIZE

class IVisualizer:

    def __init__(self):
        # self.buffer = deque(maxlen=BUFFER_SIZE)
        self.buffer = Queue.Queue(maxlen=BUFFER_SIZE)
        
    def refreshBuffer(self):
        self.buffer = Queue.Queue(maxsize=BUFFER_SIZE)
