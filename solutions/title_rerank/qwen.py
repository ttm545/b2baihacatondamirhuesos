"""Score email subjects locally with Qwen's yes/no language-model logits."""

from collections.abc import Sequence
from threading import Lock


MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = "Find email subjects that describe the specific event or document requested."
PREFIX = (
    "<|im_start|>system\nDecide whether the document matches the query and instruction. "
    "Answer only yes or no.<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class QwenSubjectScorer:
    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, padding_side="left", trust_remote_code=False,
        )
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=dtype, trust_remote_code=False, use_safetensors=True,
        ).to(self.device).eval()
        label_ids = self.tokenizer.convert_tokens_to_ids(["no", "yes"])
        with torch.no_grad():
            self.head = self.model.get_output_embeddings().weight[label_ids]
        self.prefix = self.encode(f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: ")
        self.separator = self.encode("\n<Document>: ")
        self.suffix = self.encode(SUFFIX)
        self.lock = Lock()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def score(self, query: str, titles: Sequence[str]) -> list[float]:
        if not titles:
            return []
        query_ids = self.encode(query[:1000].replace("<|", "< |"))[:64]
        start = self.prefix + query_ids + self.separator
        sequences = [start + self.encode(title.replace("<|", "< |"))[:96] + self.suffix for title in titles]
        pad = self.tokenizer.pad_token_id
        width = max(map(len, sequences))
        inputs = self.torch.tensor([[pad] * (width - len(ids)) + ids for ids in sequences], device=self.device)
        mask = self.torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences],
                                 device=self.device)
        positions = (mask.cumsum(dim=1) - 1).clamp_min_(0)
        with self.lock, self.torch.inference_mode():
            states = self.model.model(input_ids=inputs, attention_mask=mask, position_ids=positions,
                                      use_cache=False).last_hidden_state
            # The standard Qwen head is linear. Only its yes/no rows are needed here.
            logits = self.torch.nn.functional.linear(states[:, -1, :], self.head).float()
            return (logits[:, 1] - logits[:, 0]).sigmoid().tolist()
