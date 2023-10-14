import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset



def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{12288}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def create_db_from_local(dir_path):
    # define what documents to load
    loader = DirectoryLoader(dir_path, glob="*.txt", loader_cls=TextLoader)

    # interpret information in the documents
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                            chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'})

    # create and save the local database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss")


from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA


def load_llama_model(model_path):
    return CTransformers(model=model_path,
                        model_type='llama',
                        config={'max_new_tokens': 256, 'temperature': 0.01})


def init_doc_retriever():
    # load the interpreted information from the local database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'})
    db = FAISS.load_local("faiss", embeddings)

    # prepare a version of the llm pre-loaded with the local content
    return db.as_retriever(search_kwargs={'k': 2})