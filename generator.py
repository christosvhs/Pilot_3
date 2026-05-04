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

    def _build_messages(self, user_query: str, retrieved_docs: dict) -> list[dict]:
        docs = retrieved_docs.get("documents", [[]])[0]
        context = "\n\n".join(f"[Document {i + 1}]: {doc}" for i, doc in enumerate(docs))
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"},
        ]

    def generate(self, user_query: str, retrieved_docs: dict) -> str:
        messages = self._build_messages(user_query, retrieved_docs)

        print("messages: s", messages)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
