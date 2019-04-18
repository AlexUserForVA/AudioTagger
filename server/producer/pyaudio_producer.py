import time
import wave
import threading
import pyaudio
import numpy as np

from server.config.config import BUFFER_SIZE

class MicrophoneThread(threading.Thread):

    CHUNK_SIZE = 1024
    SAMPLE_RATE = 32000

    def __init__(self, model, name='MicrophoneThread'):
        self.p = pyaudio.PyAudio()
        self.model = model
        self._stopevent = threading.Event()
        threading.Thread.__init__(self, name=name)

    def run(self):
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=self.SAMPLE_RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK_SIZE)

        while not self._stopevent.isSet():
            chunk = stream.read(self.CHUNK_SIZE)
            chunk = np.fromstring(chunk, np.int16)
            self.model.putToSM(chunk)


    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        threading.Thread.join(self, timeout)


class FileThread(threading.Thread):

    CHUNK_SIZE = 1024

    def __init__(self, model, filePath, name='FileThread'):
        self.wf = wave.open(filePath, 'rb')
        self.t = 0
        self.model = model
        self.filePath = filePath
        self._stopevent = threading.Event()
        threading.Thread.__init__(self, name=name)

    def run(self):
        data = self.wf.readframes(self.CHUNK_SIZE)
        while not self._stopevent.isSet() and data != b'':
            time.sleep(0.03)
            # print(time.time())
            self.model.putToSM(data)
            data = self.wf.readframes(self.CHUNK_SIZE)
            self.t += 1
            self.t = self.t % BUFFER_SIZE

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        threading.Thread.join(self, timeout)