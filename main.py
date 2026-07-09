from langchain_community.document_loaders import TextLoader
loader = TextLoader("./documents/nvda_news_1.txt")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# from 
# loading file
data = loader.load()
# text splitters
splitter = RecursiveCharacterTextSplitter(
  chunk_size = 500,
  chunk_overlap = 100,
)
chunks = splitter.split_documents(data)
# print(len(chunks))
# print(chunks[1].page_content)


# create embeddings
embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# embedding all chunks
texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(texts)

# storing in vector db
db = FAISS.from_documents(
  documents = chunks,
  embedding = embeddings
)
db.save_local("faiss_index")

# results = db.similarity_search(query,k=3)
# for doc in results : 
#   print(doc.page_content)
#   print("-"*50)

# intializing the llm
llm = ChatGroq(
  model = "llama-3.3-70b-versatile",
  temperature=0
)
# creating the retreiver
retriever = db.as_retriever(
  search_kwargs = {"k" : 3}
)
# retrieve the documents

query = "What is NVIDIA's financial strength ? "
docs = retriever.invoke(query)
for doc in docs : 
  print(doc.page_content)


# send retrieved content to the llm
context = "\n\n".join([doc.page_content for doc in docs])
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from yaspin import yaspin

console = Console()

console.clear()

console.print(
    Panel.fit(
        "[bold cyan] Local RAG Assistant[/bold cyan]\n"
        "[green]Powered by FAISS + HuggingFace + Groq[/green]",
        border_style="bright_blue",
    )
)

console.print("[yellow]Type 'exit' or 'quit' to leave.[/yellow]\n")

while True:

    query = console.input("[bold cyan]You ❯ [/]").strip()

    if query.lower() in ("exit", "quit"):
        console.print("\n[bold red]👋 Goodbye![/bold red]")
        break

    # Retrieve documents
    with yaspin(text="Retrieving relevant documents...", color="cyan") as spinner:
        docs = retriever.invoke(query)
        spinner.ok("✅")

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
reply with:

I don't know based on the provided context.

Context:
{context}

Question:
{query}
"""

    # LLM generation
    with yaspin(text="Generating answer...", color="green") as spinner:
        response = llm.invoke(prompt)
        spinner.ok("🤖")

    console.print()

    console.print(
        Panel(
            Markdown(response.content),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )
    )

    console.print(Rule("[bold blue]Retrieved Context[/bold blue]"))

    for i, doc in enumerate(docs, start=1):
        preview = doc.page_content.replace("\n", " ")

        if len(preview) > 180:
            preview = preview[:180] + "..."

        console.print(
            Panel(
                preview,
                title=f"Chunk {i}",
                border_style="cyan",
            )
        )