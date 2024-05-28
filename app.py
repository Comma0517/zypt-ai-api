import os
from flask import Flask, request, Response, stream_with_context, jsonify
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from docx import Document
from openpyxl import load_workbook

from langchain.prompts import PromptTemplate
from langchain.chains import (
    LLMChain,
    ConversationalRetrievalChain,
)
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from pymongo import MongoClient

load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

app = Flask(__name__)

# this will need to be reconfigured before taking the app to production
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/chat_gpt", methods=["POST"])
@cross_origin()
def chat():
    try:
        body = request.json

        template = """You're a helpful AI assistant tasked to answer the user's questions.
        You're friendly and you answer extensively with multiple sentences. You prefer to use bullet-points to summarize.

        CHAT HISTORY: 
        {chat_history}

        QUESTION:
        {question}

        YOUR ANSWER:"""

        system_prompt_template = """You're a helpful AI assistant tasked to answer the user's questions.  
        If you don't know the answer, don't say that you don't know, try to make up an answer abundantly.
        But You have to follow bellow SYSTEM PROMPT

        SYSTEM PROMPT: {system_prompt}

        CONTEXT: {context} 

        QUESTION: {question}

        YOUR ANSWER:"""

        formatted_template = system_prompt_template.format(
            system_prompt=body["system_prompt"],
            context="{context}",
            question="{question}"
        )

        QA_PROMPT = PromptTemplate(
            template=formatted_template, input_variables=["system_prompt", "context", "question"]
        )

        prompt = PromptTemplate.from_template(template)

        llm = AzureChatOpenAI(
            openai_api_version="2023-12-01-preview",  # e.g., "2023-12-01-preview"
            azure_deployment="gpt",
            temperature=0,
        )

        memory = ConversationSummaryBufferMemory(
            memory_key="chat_history",
            llm=llm,
            # max_token_limit=300,
            return_messages=True,
            input_key="question",
        )

        aoai_embeddings = AzureOpenAIEmbeddings(
            azure_deployment="embedding",
            openai_api_version="2023-12-01-preview",  # e.g., "2023-12-01-preview"
            chunk_size=10
        )

        client: MongoClient = MongoClient(CONNECTION_STRING)
        collection = client['mydatabase']['mycontainer']

        vectorstore = AzureCosmosDBVectorSearch(
            collection, aoai_embeddings, index_name='vectorSearchIndex'
        )

        retriever = vectorstore.as_retriever()

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            condense_question_prompt=prompt,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        response = qa(
            {
                "chat_history": "",
                "question": body["human_input"],
            }
        )
        print(response)
        return jsonify({"message": "Success", "llm_response": response["answer"]}), 200
    except Exception as e:
        return jsonify({"error": "{}: {}".format(type(e).__name__, str(e))}), 500


@app.route("/")
def hello_world():  # put application's code here
    return "Hello World!"


if __name__ == "__main__":
    app.run()
