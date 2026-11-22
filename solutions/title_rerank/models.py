"""Available local subject scorers and their tuned weights."""

from .jina import JinaSubjectScorer
from .qwen import QwenSubjectScorer


MODELS = {
    "jina": (JinaSubjectScorer, 0.35),
    "qwen": (QwenSubjectScorer, 0.35),
}
