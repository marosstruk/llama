from langchain.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

import os
import json
import time
import logging
from shutil import rmtree
from typing import Dict


class Llama:
    
    TEMP_DIR = "./tmp"
    
   
    def __init__(self):
        # prepare the template we will use when prompting the AI
        template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

        # load the language model
        self.llm = CTransformers(model='./models/llama-2-13b-chat.ggmlv3.q4_1.bin',
                            model_type='llama',
                            config={'max_new_tokens': 256, 'temperature': 0.01})

        # load the interpreted information from the local database
        # embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",
        #     model_kwargs={'device': 'cpu'})
        # db = FAISS.load_local("faiss", embeddings)
    
        self.promptTemplate = PromptTemplate(
            template=template,
            input_variables=['context', 'question'])
    

    def read_documents(self, docs: list):
        start_time = time.time()
        tmp_dir = os.path.join(self.TEMP_DIR, str(round(time.time())))
        os.mkdir(tmp_dir)
        
        for doc in docs:
            doc_id = str(doc["_id"])
            tmp_file_path = os.path.join(tmp_dir, f"{doc_id}.json")
            
            with open(tmp_file_path, "w") as tmp_file:
                # combined_text = "\n\n".join([page["Text"] for page in doc["Data"]["Pages"]])
                # tmp_file.write(combined_text)
                tmp_file.write(json.dumps(doc["Data"]))
        
        # define what documents to load
        loader = DirectoryLoader(tmp_dir, glob="*.json", loader_cls=JSONLoader,
                                 loader_kwargs={"jq_schema": ".Pages[].Text"})
                                 #use_multithreading=True, max_concurrency=16)
        
        # interpret information in the documents
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={'k': 2})
        
        # Put the llm and document store together, ready for prompts
        self.qa_llm = RetrievalQA.from_chain_type(llm=self.llm,
                        chain_type='stuff',
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': self.promptTemplate})
        
        end_time = time.time()
        elapsed_time = round((end_time - start_time) * 1000, 2)
        logging.info(f"Read {len(docs)} documents in {elapsed_time} ms")
        
        # Clean up temporary files
        rmtree(tmp_dir, onerror=lambda err:
            logging.warning(f"Failed to remove temporary files: {tmp_dir} Error: {err}"))


    def answer_prompt(self, prompt):
        output = self.qa_llm({'query': prompt})
        return output


from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from bson import ObjectId, json_util

if __name__ == "__main__":
    llm = Llama()
    
    conn_str = os.environ.get("MONGO_CONN_STR", "mongodb://172.26.0.1:27017")
    client = MongoClient(conn_str)
    db: Database = client.docparser
    coll: Collection = db.docs
    
    docs = list(coll.find())
    
    llm.read_documents(docs)