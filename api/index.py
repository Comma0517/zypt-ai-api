import os
import asyncio
from dotenv import load_dotenv
from quart import Quart, request, jsonify
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity, ActivityTypes
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from pymongo import MongoClient
import logging

load_dotenv()

# Environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
MICROSOFT_APP_ID = os.getenv("MICROSOFT_APP_ID")
MICROSOFT_APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, CONNECTION_STRING, MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD]):
    raise ValueError("One or more environment variables are missing")

# Bot and adapter settings
adapter_settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

# Azure LLM and embeddings setup
llm = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",
    azure_deployment="gpt",
    temperature=0,
)

aoai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding",
    openai_api_version="2023-12-01-preview",
    chunk_size=10
)

client = MongoClient(CONNECTION_STRING)
collection = client['mydatabase']['mycontainer']

vectorstore = AzureCosmosDBVectorSearch(
    collection, aoai_embeddings, index_name='vectorSearchIndex'
)

retriever = vectorstore.as_retriever()

# Quart app for bot adapter
app = Quart(__name__)

async def process_message_async(human_input):
    try:
        logging.info(f"Processing message: {human_input}")

        template = """
        You're a helpful AI assistant tasked to answer the user's questions.
        You're friendly and you answer extensively with multiple sentences. You prefer to use bullet-points to summarize.
        
        CHAT HISTORY: 
        {chat_history}
        
        QUESTION:
        {question}
        
        YOUR ANSWER:
        """

        system_prompt_template = """
        You're a helpful AI assistant tasked to answer the user's questions.  
        If you don't know the answer, don't say that you don't know, try to make up an answer abundantly.
        But You have to follow below SYSTEM PROMPT
        
        SYSTEM PROMPT: {system_prompt}
        
        CONTEXT: {context} 
        
        QUESTION: {question}
        
        YOUR ANSWER:
        """

        formatted_template = system_prompt_template.format(
            system_prompt=human_input["system_prompt"],
            context="{context}",
            question="{question}"
        )

        QA_PROMPT = PromptTemplate(
            template=formatted_template, input_variables=["system_prompt", "context", "question"]
        )

        prompt = PromptTemplate.from_template(template)

        memory = ConversationSummaryBufferMemory(
            memory_key="chat_history",
            llm=llm,
            return_messages=True,
            input_key="question",
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            condense_question_prompt=prompt,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        response = await qa.invoke_async({
            "chat_history": "",
            "question": human_input["human_input"],
        })
        
        logging.info(f"Generated response: {response['answer']}")
        return response["answer"]
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        return "Sorry, I encountered an error processing your request."

@app.route("/api/message", methods=["POST"])  # Note the singular 'message'
async def message():
    try:
        body = await request.get_json()
        logging.info(f"Received request body: {body}")
        activity = Activity().deserialize(body)
        logging.info(f"Deserialized activity: {activity}")
        auth_header = request.headers.get("Authorization", "")

        if activity.type == ActivityTypes.message:
            async def aux(turn_context: TurnContext):
                user_input = activity.text
                human_input = {"human_input": user_input, "system_prompt": ""}
                response = await process_message_async(human_input)
                await turn_context.send_activity(response)
            await adapter.process_activity(activity, auth_header, aux)
        elif activity.type == ActivityTypes.conversation_update:
            logging.info("Handling conversation update activity")
            async def aux(turn_context: TurnContext):
                if activity.members_added:
                    for member in activity.members_added:
                        if member.id != activity.recipient.id:
                            await turn_context.send_activity(f"Welcome {member.name or ''}!")
            await adapter.process_activity(activity, auth_header, aux)
        else:
            raise TypeError("Invalid activity type or missing activity")

        return jsonify({"status": "success"}), 201
    except Exception as e:
        logging.error(f"Error handling request: {e}")
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/")
async def health_check():
    return "Hello World!"

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    app.run(host="0.0.0.0", port=8000)