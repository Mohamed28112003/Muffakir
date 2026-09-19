from .models import QAPair, LLMQAOutput, GenerationStats, SyntheticDataConfig

__all__ = [
    "QAPair",
    "LLMQAOutput",
    "GenerationStats",
    "SyntheticDataConfig",
    "QAParser",
    "QAGenerator",
    "DatasetExporter",
    "SyntheticDataPipeline",
]


def __getattr__(name: str):
    if name == "QAParser":
        from .parser import QAParser
        return QAParser
    if name == "QAGenerator":
        from .generator import QAGenerator
        return QAGenerator
    if name == "DatasetExporter":
        from .exporter import DatasetExporter
        return DatasetExporter
    if name == "SyntheticDataPipeline":
        from .pipeline import SyntheticDataPipeline
        return SyntheticDataPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
