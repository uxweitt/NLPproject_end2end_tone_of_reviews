import pytest

from final.ml.model_app import SentimentPrediction, load_model


@pytest.fixture(scope="function")
def model():
    return load_model()

@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("Фильм полный отстой, конкретный ватафа", "neg"),
        ("Фильм приятный, мне понравился!", "pos"),
        ("Фильм уровня Данила Колбасенко и Кузи Лакомкина оказался неплох", "neu"),
    ],
)
def test_sentiment(model, text, expected_label):
    model_pred = model(text)
    assert isinstance(model_pred, SentimentPrediction)
    assert model_pred.label == expected_label