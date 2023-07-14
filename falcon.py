#!/usr/bin/env python3
import os
import sys
import time
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All

app = Flask(__name__)
CORS(app)
load_dotenv()

# Load environment variables
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

qa = None
db = None
retriever = None
embeddings = None
args = None
callbacks = None


def main():
    global args, embeddings, db, retriever, callbacks, qa
    args = parse_arguments()
    print(args)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Initialize Chroma database
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

    # Initialize retriever
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Prepare the LLM
    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj',
                  n_batch=model_n_batch, callbacks=callbacks, verbose=False)

    # Initialize the RetrievalQA instance
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)

    print("Model Loaded")


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')

    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

def Chat(user_input):
    print("User Input", user_input)
    if user_input.strip() == "":
        return "Invalid input","No Docs"
    start = time.time()
    res = qa(user_input)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']
    end = time.time()
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    return answer,docs

@app.route("/answer", methods=["POST"])
@cross_origin()
def answer_question():
    user_input=request.json["question"]
    bot_answer,docs=Chat(user_input)
    response = {
        "answer": bot_answer,
        "documents": [doc.page_content for doc in docs]
    }
    return jsonify(response)

if __name__ == "__main__":
    main()
    if True:
        print("-----------------------------")
        print("Welcome to terminal chat\nType exit to exit chat")

        continue_chat=True
        while continue_chat:
            user_input=input("Query : ")

            if user_input == "exit":
                print("--Bye--")
                break

            bot_answer,docs=Chat(user_input)
            print("BOT : "+bot_answer)
    else:
        app.run(host='0.0.0.0', port=8887)


    
