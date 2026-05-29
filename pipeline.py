import threading

from retriever import Retriever
from generator import Generator, AVAILABLE_MODELS


class RAGPipeline:
    def __init__(
        self,
        n_results: int = 20,
        default_llm: str = "salamandra",
        load_in_4bit: bool = False,
    ):
        self.retrievers = {
            "finetuned": Retriever(n_results=n_results, use_finetuned=True),
            "baseline": Retriever(n_results=n_results, use_finetuned=False),
        }
        self.load_in_4bit = load_in_4bit
        self.current_llm = default_llm
        self.generator = Generator(
            model_name=AVAILABLE_MODELS[default_llm],
            load_in_4bit=load_in_4bit,
        )
        self._lock = threading.Lock()

    def _switch_llm_unlocked(self, llm_type: str):
        if llm_type == self.current_llm or llm_type not in AVAILABLE_MODELS:
            return
        print(f"\n[PIPELINE] Switching LLM: {self.current_llm} -> {llm_type}")
        self.generator.unload()
        self.generator = Generator(
            model_name=AVAILABLE_MODELS[llm_type],
            load_in_4bit=self.load_in_4bit,
        )
        self.current_llm = llm_type

    def switch_llm(self, llm_type: str):
        with self._lock:
            self._switch_llm_unlocked(llm_type)

    def run(
        self,
        dialog_history: list[str],
        retriever_type: str = "finetuned",
        llm_type: str = "salamandra",
    ) -> dict:
        with self._lock:
            self._switch_llm_unlocked(llm_type)
            retriever = self.retrievers.get(retriever_type, self.retrievers["finetuned"])
            user_query = dialog_history[-1]
            retrieved = retriever.retrieve(dialog_history)
            response = self.generator.generate(dialog_history, retrieved)
            return {
                "query": user_query,
                "retrieved_documents": retrieved,
                "response": response,
            }
