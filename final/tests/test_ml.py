import pytest

from ml.model import SentimentPrediction, load_model


@pytest.fixture(scope="function")
def model():
    return load_model()

@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("бэд", "neg"),
        ("гуд", "pos"),
        ("нот бэд", "neu"),
    ],
)
def test_sentiment(model, text, expected_label):
    model_pred = model(text)
    assert isinstance(model_pred, SentimentPrediction)
    assert model_pred.label == expected_label