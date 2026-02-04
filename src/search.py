import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda EXATAMENTE:
    "Não tenho informações necessárias para responder sua pergunta."
- Nunca adicione explicações ou justificativas à resposta padrão acima.
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL"),
    model_kwargs={"device": "cpu"},
)

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

def search_context(question: str, k: int = 10) -> list:
    """Search for relevant documents based on the question."""
    results = store.similarity_search_with_score(question, k=k)
    return results

def build_prompt(question: str, k: int = 10) -> str:
    """Build the full prompt with context from vector search."""
    docs = search_context(question, k=k)
    contexto = "\n\n".join([doc.page_content for doc, score in docs])
    return PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)