from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate, Settings

from rag.document_loader import load_index
from rag.model_engine import ModelEngine

Settings.llm = ModelEngine()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", trust_remote_code=True, device="cuda")

data_dir = "data/rag"
index = load_index(data_dir)

qa_template = PromptTemplate((
    "Below is an instruction that describes a task. Write a response that fulfills the task. \n"
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "### Instruction:\n"
    "Given this information, please answer the question\n"
    "### Input:\n"
    "{query_str}\n"
))
query_engine = index.as_query_engine(similarity_top_k=2)
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_template}
)

response = query_engine.query("Answer the following question: Which company creates MyPro?")
print(response)