from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os

load_dotenv()

llama_embeddings_model = os.getenv("LLAMA_EMBEDDINGS_MODEL")
persist_directory = os.getenv('PERSIST_DIRECTORY')
model_type = os.getenv('MODEL_TYPE')
model_path = os.getenv('MODEL_PATH')
model_n_ctx = os.getenv('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

def main(doc_limit=1000):
    if not all([llama_embeddings_model, persist_directory, model_type, model_path, model_n_ctx]):
        print("Error: One or more environment variables are not set.")
        return

    try:
        llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    except Exception as e:
        print(f"Error initializing LlamaCppEmbeddings: {e}")
        return

    try:
        db = Chroma(persist_directory=persist_directory, embedding_function=llama, client_settings=CHROMA_SETTINGS)
    except Exception as e:
        print(f"Error initializing Chroma: {e}")
        return

    retriever = db.as_retriever()

    callbacks = [StreamingStdOutCallbackHandler()]
    
    try:
        match model_type:
            case "LlamaCpp":
                llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
            case "GPT4All":
                llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
            case _default:
                print(f"Model {model_type} not supported!")
                return
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    doc_count = 0
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
            
            doc_count += 1
            if doc_count >= doc_limit:
                print(f"\nDocument limit of {doc_limit} reached.")
                break

if __name__ == "__main__":
    main()
