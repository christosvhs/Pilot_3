from retriever import Retriever, DEFAULT_MODEL_DIR
from generator import Generator, DEFAULT_MODEL


class RAGPipeline:
    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        n_results: int = 10,
        llm_model: str = DEFAULT_MODEL,
        load_in_4bit: bool = False,
    ):
        self.retriever = Retriever(model_dir=model_dir, n_results=n_results)
        self.generator = Generator(model_name=llm_model, load_in_4bit=load_in_4bit)

    def run(self, dialog_history: list[str]) -> dict:
        user_query = dialog_history[-1]
        retrieved = self.retriever.retrieve(dialog_history)
        response = self.generator.generate(dialog_history, retrieved)
        return {
            "query": user_query,
            "retrieved_documents": retrieved,
            "response": response,
        }
