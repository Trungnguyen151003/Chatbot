import os
import glob
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Annotated
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import AsyncCallbackHandler

class BufferedWebSocketHandler(AsyncCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.buffer = ""

    async def on_llm_new_token(self, token: str, **kwargs):
        self.buffer += token
        # Send token if it completes a word/punctuation (optional tweak)
        if token.strip().endswith((" ", ".", ",", "!", "?", ";", ":", "\n")):
            await self.websocket.send_text(self.buffer)
            self.buffer = ""

    async def on_llm_end(self, response, **kwargs):
        # Ensure remaining buffer content is sent
        if self.buffer:
            await self.websocket.send_text(self.buffer)
            self.buffer = ""



app = FastAPI()
templates = Jinja2Templates(directory="templates")

load_dotenv(override=True)

# OpenAI setup
MODEL = "gpt-4o-mini"
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

# Load and process documents
folders = glob.glob("knowledge-base/*")
text_loader_kwargs = {'encoding': 'utf-8'}
documents = []

for folder in folders:
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create or load Chroma vector store
db_name = "vector_db"
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

# Setup conversation chain
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
llm = ChatOpenAI(temperature=0.6, model_name=MODEL)
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "chat_responses": []})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            user_input = await websocket.receive_text()

            # Initialize buffered streaming handler
            stream_handler = BufferedWebSocketHandler(websocket)

            # Streaming LLM with buffered callback
            streaming_llm = ChatOpenAI(
                temperature=0.6,
                model_name=MODEL,
                streaming=True,
                callbacks=[stream_handler]
            )

            # Streaming conversation chain
            streaming_chain = ConversationalRetrievalChain.from_llm(
                llm=streaming_llm,
                retriever=retriever,
                memory=memory,
                output_key='answer'
            )

            # Execute chain (response buffered & streamed)
            await streaming_chain.ainvoke({"question": user_input})

        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")
            break


@app.post("/", response_class=HTMLResponse)
async def chat(request: Request, user_input: Annotated[str, Form()]):
    result = conversation_chain.invoke({"question": user_input})
    bot_response = result["answer"]
    return templates.TemplateResponse("home.html", {"request": request, "chat_responses": [user_input, bot_response]})

@app.get("/image", response_class=HTMLResponse)
async def image_page(request: Request):
    return templates.TemplateResponse("image.html", {"request": request})

@app.post("/image", response_class=HTMLResponse)
async def generate_image(request: Request, user_input: Annotated[str, Form()]):
    response = llm.client.images.generate(prompt=user_input, n=1, size='256x256')
    image_url = response.data[0].url
    return templates.TemplateResponse("image.html", {"request": request, "image_url": image_url})