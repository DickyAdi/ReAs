from domain.io.interfaces import StreamerInterface


class JsonStreamerApplication:
    def __init__(self, service: StreamerInterface):
        self.service = service

    def get_streamer(self):
        return self.service
