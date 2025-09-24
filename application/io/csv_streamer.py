from domain.io.interfaces import StreamerInterface


class ReviewCsvStreamerApplication:
    def __init__(self, service: StreamerInterface):
        """Initializing csv streamer for review data.

        Args:
            service (StreamerInterface): Initialized ReviewCsvStreamer with correct `file_path`, `text_column`, `rating_column`, and `batch_size`
        """
        self.service = service

    def get_streamer(self):
        return self.service
