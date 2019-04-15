import threading

class ProducerThread(threading.Thread):

    def __init__(self, model, audioSource, name='ProducerThread'):
        self.model = model
        self.audioSource = audioSource
        self._stopevent = threading.Event()
        threading.Thread.__init__(self, name=name)

    def run(self):
        stream = self.model.producer.produce(self.audioSource)
        for frame in stream:
            if self._stopevent.isSet():
                return
            self.model.put_signal(frame)

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        threading.Thread.join(self, timeout)

class SignalProvider:

    # function stems from madmom (https://github.com/CPJKU/madmom/blob/master/madmom/processors.py)
    def produce(self, infile, **kwargs):
        """
        Process a file or audio stream with the given Processor.
        Parameters
        ----------
        processor : :class:`Processor` instance
            Processor to be processed.
        infile : str or file handle, optional
            Input file (handle). If none is given, the stream present at the
            system's audio input is used. Additional keyword arguments can be used
            to influence the frame size and hop size.
        outfile : str or file handle
            Output file (handle).
        kwargs : dict, optional
            Keyword arguments passed to :class:`.audio.signal.Stream` if
            `in_stream` is 'None'.
        Notes
        -----
        Right now there is no way to determine if a processor is online-capable or
        not. Thus, calling any processor with this function may not produce the
        results expected.
        """
        from madmom.audio.signal import Stream, FramedSignal
        global counter
        # set default values
        # kwargs['sample_rate'] = kwargs.get('sample_rate', 44100)
        kwargs['sample_rate'] = kwargs.get('sample_rate', 32000)
        kwargs['num_channels'] = kwargs.get('num_channels', 1)
        kwargs['fps'] = kwargs.get('fps', 32000.0 / 441.0)

        # if no input file is given, create a Stream with the given arguments
        if infile is None:
            # open a stream and start if not running already
            stream = Stream(**kwargs)
            if not stream.is_running():
                stream.start()
        # use the input file
        else:
            # set parameters for opening the file
            from madmom.audio.signal import FRAME_SIZE, HOP_SIZE, FPS, NUM_CHANNELS
            # frame_size = 1024
            # hop_size = 128
            # num_channels = 1
            # fps = None
            frame_size = kwargs.get('frame_size', FRAME_SIZE)
            hop_size = kwargs.get('hop_size', HOP_SIZE)
            fps = kwargs.get('fps', FPS)
            num_channels = kwargs.get('num_channels', NUM_CHANNELS)
            # FIXME: overwrite the frame size with the maximum value of all used
            #        processors. This is needed if multiple frame sizes are used
            import warnings
            warnings.warn('make sure that the `frame_size` (%d) is equal to the '
                          'maximum value used by any `FramedSignalProcessor`.' %
                          frame_size)
            # Note: origin must be 'online' and num_frames 'None' to behave exactly
            #       the same as with live input
            stream = FramedSignal(infile, frame_size=frame_size, hop_size=hop_size,
                                  fps=fps, origin='online', num_frames=None,
                                  num_channels=num_channels, sample_rate=kwargs['sample_rate'])
        # set arguments for online processing
        # Note: pass only certain arguments, because these will be passed to the
        #       processors at every time step (kwargs contains file handles etc.)
        # process_args = {'reset': False}  # do not reset stateful processors
        # process everything frame-by-frame
        return stream


    def registerModel(self, model):
        self.model = model


