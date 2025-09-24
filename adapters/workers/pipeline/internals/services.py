from typing import BinaryIO, Optional, Union

from application.io import ReviewCsvStreamerApplication
from infrastructure.io.csv_review_streamer import ReviewCsvStreamer


class PipelineService:
    def __init__(
        self,
        # file: BinaryIO,
        data: Union[BinaryIO, str],
        text_column: str,
        rating_column: Optional[str] = None,
        batch_size: Optional[int] = 512,
    ):
        # self.file = file
        self.data = data
        self.text_column = text_column
        self.rating_column = rating_column
        self.batch_size = batch_size
        self.csv_streamer = ReviewCsvStreamerApplication(
            service=ReviewCsvStreamer(
                # file=self.file,
                file=self.data,
                text_column=self.text_column,
                rating_column=self.rating_column,
                batch_size=self.batch_size,
            )
        ).get_streamer()
        self.json_streamer = None  # later do .get_streamer()
        self.streamer = (
            self.csv_streamer if isinstance(self.data, BinaryIO) else self.json_streamer
        )

    def review_split(self):
        reviews = []
        with self.streamer as s:
            while True:
                review = s.next_batch()
                if not review:
                    return reviews
                reviews.append(review)

    def get_streamer(self):
        return self.streamer
