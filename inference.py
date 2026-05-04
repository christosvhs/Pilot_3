from pipeline import RAGPipeline

pipeline = RAGPipeline(n_results=10)

# dialog_history: alternating system/user turns; last element is always the user query
dialog_history = [
    "Hello",
    "Hello! How can I assist you today?",
    "What are supplementary policy benefits and are they included in a conversion policy?",
]

result = pipeline.run(dialog_history)

print("Query:", result["query"])
print("\nTop retrieved documents:")
for i, doc in enumerate(result["retrieved_documents"]["documents"][0]):
    print(f"  [{i + 1}] {doc}")
print("\nGenerated response:\n", result["response"])
