import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEFAULT_MODEL = "BSC-LT/salamandra-7b-instruct"

AVAILABLE_MODELS = {
    "salamandra": "BSC-LT/salamandra-7b-instruct",
    "krikri": "ilsp/Llama-Krikri-8B-Instruct",
}

_SYSTEM_PROMPT = (
    "You are a customer service assistant for a call center. You will receive background "
    "information retrieved from the company knowledge base, followed by the conversation "
    "with the user.\n\n"
    "CRITICAL: The background information is your ONLY source of facts. You have no "
    "general knowledge. Even if you know an answer from training, you MUST NOT use it. "
    "A passing mention of a term in the background does not license you to elaborate on "
    "it from outside knowledge — the answer must be explicitly stated.\n\n"
    "Rules:\n"
    "1. If the user's question is unrelated to customer service (e.g. small talk, jokes, "
    "math, code, general trivia), politely tell them you can only help with customer "
    "service inquiries.\n"
    "2. If the background information explicitly answers the question, respond directly "
    "and naturally. Do NOT reference the source — never say things like \"according to "
    "the document\", \"based on the context\", or \"the information provided says\". "
    "Answer as if you already knew it.\n"
    "3. If the background information does not explicitly answer the question — even if "
    "you know the answer from general knowledge — you MUST say you do not have that "
    "information. Never guess, infer beyond what is stated, or fill gaps with outside "
    "knowledge.\n"
    "4. Reply in the same language the user uses.\n"
    "5. Be concise and direct."
)


class Generator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 512,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self):
        print(f"\n[GENERATOR] Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        print(f"[GENERATOR] Loaded: {self.model_name}")

    def unload(self):
        print(f"\n[GENERATOR] Unloading model: {self.model_name}")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
