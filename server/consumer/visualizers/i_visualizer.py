from collections import deque

from server.config.config import BUFFER_SIZE

class IVisualizer:

    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)