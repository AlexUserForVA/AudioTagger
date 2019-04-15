import queue
# from collections import deque

from server.config.config import BUFFER_SIZE

class IVisualizer:
        
    def refreshBuffer(self):
        self.buffer = queue.Queue(maxsize=BUFFER_SIZE)
