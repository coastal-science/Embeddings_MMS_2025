"""Smoke test asserting the preprocessor package imports cleanly."""


def test_import_preprocessor():
    import encoder_pipeline.preprocessor.annotation  # noqa: F401
    import encoder_pipeline.preprocessor.config  # noqa: F401
    import encoder_pipeline.preprocessor.dataset  # noqa: F401
    import encoder_pipeline.preprocessor.spectrogram  # noqa: F401
