from fastapi.testclient import TestClient
import io

from main import app
from services import startup

class DummyModel:
    def __init__(self):
        self.n = 1
    def predict(self, text):
        if self.n % 2 == 0:
            self.n += 1
            return 'Positive', None
        else:
            self.n += 1
            return 'Negative', None

def test_main(monkeypatch):
    # monkeypatch.setattr(startup, "load_model", lambda: DummyModel())

    content = b"review\nGreat place\nNice coffee"

    with TestClient(app) as client:
        client.app.state.model = DummyModel()
        response = client.post(
            "/extract?text_column=review",
            files={'file' : ('test.csv', io.BytesIO(content), 'text/csv')}
            )
        assert response.status_code == 200
        assert 'positive_topics' in response.json()['data']
