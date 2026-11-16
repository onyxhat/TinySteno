"""The parts of sherpa-onnx's Python API that tinysteno.sherpa_diarizer uses.

Verified against sherpa-onnx 2.0.1. No model is loaded and no diarizer is run.
"""
import pytest

sherpa_onnx = pytest.importorskip("sherpa_onnx")


@pytest.mark.parametrize("name", [
    "OfflineSpeakerDiarization", "OfflineSpeakerDiarizationConfig",
    "OfflineSpeakerSegmentationModelConfig", "OfflineSpeakerSegmentationPyannoteModelConfig",
    "SpeakerEmbeddingExtractorConfig", "FastClusteringConfig",
])
def test_class_exists(name):
    assert hasattr(sherpa_onnx, name)


def make_config():
    """Build a config with every field the plan sets, pointing at files that do not exist."""
    return sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model="seg.onnx"),
            num_threads=2),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model="emb.onnx", num_threads=2),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.6),
        min_duration_on=0.2, min_duration_off=0.5,
    )


def test_config_accepts_the_fields_we_set():
    config = make_config()
    assert config.clustering.num_clusters == -1
    assert config.clustering.threshold == pytest.approx(0.6)
    assert config.embedding.model == "emb.onnx"
    assert config.embedding.num_threads == 2
    assert config.segmentation.pyannote.model == "seg.onnx"
    assert config.segmentation.num_threads == 2
    assert config.min_duration_on == pytest.approx(0.2)
    assert config.min_duration_off == pytest.approx(0.5)


def test_config_with_missing_model_files_does_not_validate():
    # validate() returns False (and logs to stderr) rather than raising.
    assert make_config().validate() is False


def test_diarizer_with_missing_model_files_raises_runtime_error():
    with pytest.raises(RuntimeError):
        sherpa_onnx.OfflineSpeakerDiarization(make_config())


def test_diarizer_surface_we_call():
    diarizer = sherpa_onnx.OfflineSpeakerDiarization
    assert isinstance(diarizer.sample_rate, property)
    # process(samples, callback=None); callback(processed_chunks, num_chunks) -> int
    assert "callback" in diarizer.process.__doc__
    result = sherpa_onnx.OfflineSpeakerDiarizationResult
    assert callable(result.sort_by_start_time)
    assert callable(result.sort_by_speaker)
    segment = sherpa_onnx.OfflineSpeakerDiarizationSegment
    for attribute in ("start", "end", "duration", "speaker"):
        assert hasattr(segment, attribute)
