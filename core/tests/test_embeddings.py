from core.embeddings import embed, cosine, to_blob, from_blob


def test_similar_beats_unrelated():
    a = embed("Remind me to call Ravi about the demo.")
    b = embed("I need to phone Ravi regarding the presentation.")
    c = embed("Buy vegetables from the market tomorrow.")
    assert cosine(a, b) > cosine(a, c)


def test_blob_roundtrip():
    v = embed("hello world")
    assert (from_blob(to_blob(v)) == v).all()