import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEFAULT_MODEL = "BSC-LT/salamandra-7b-instruct"

_SYSTEM_PROMPT = (
    "You are a helpful multilingual assistant. Answer the user's question using only the provided context. "
    "If the context does not contain enough information to answer, say so clearly."
)


class Generator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 512,
        load_in_4bit: bool = False,
    ):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def _build_messages(self, dialog_history: list[str], retrieved_docs: dict) -> list[dict]:
        docs = retrieved_docs.get("documents", [[]])[0]
        context = "\n\n".join(f"[Document {i + 1}]: {doc}" for i, doc in enumerate(docs))

        # inject context into system prompt so it's available across all turns
        system = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}"
        messages = [{"role": "system", "content": system}]

        # rebuild conversation turns (dialog_history alternates user/assistant)
        roles = ["user", "assistant"]
        for i, turn in enumerate(dialog_history):
            messages.append({"role": roles[i % 2], "content": turn})

        return messages

    def generate(self, dialog_history: list[str], retrieved_docs: dict) -> str:
        messages = self._build_messages(dialog_history, retrieved_docs)

        print(f"\n[GENERATOR] Sending {len(messages) - 1} turns to LLM (excl. system message)")

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
