from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import torch
import os

EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2" # Chroma takes care if embeddings are None
EMB_SBERT_MINILM = "sentence-transformers/all-MiniLM-L6-v2" # Chroma takes care if embeddings are None
LLM_FALCON_SMALL = "falcon-7b-instruct"

class PdfQA:
    def __init__(self,config:dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None
    
    @classmethod
    def create_sbert_mpnet(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})    
    
    @classmethod
    def create_falcon_instruct_small(cls, load_in_8bit=False):
        model = "falcon-7b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                tokenizer = tokenizer,
                trust_remote_code = True,
                max_new_tokens=100,
                model_kwargs={
                    "device_map": "auto", 
                    "load_in_8bit": load_in_8bit, 
                    "max_length": 512, 
                    "temperature": 0.01,
                    "torch_dtype":torch.bfloat16,
                    }
            )
        return hf_pipeline
    
    def init_embeddings(self) -> None:
        if self.config["embedding"] == EMB_SBERT_MPNET_BASE:
            if self.embedding is None:
                self.embedding = PdfQA.create_sbert_mpnet()
        else:
            self.embedding = None

    def init_models(self) -> None:
        load_in_8bit = self.config.get("load_in_8bit",False)
        if self.config["llm"] == LLM_FALCON_SMALL:
            if self.llm is None:
                self.llm = PdfQA.create_falcon_instruct_small(load_in_8bit=load_in_8bit)
        else:
            raise ValueError("Invalid config")   
             
    def vector_db_pdf(self) -> None:
        pdf_path = self.config.get("pdf_path",None)
        persist_directory = self.config.get("persist_directory",None)
        if persist_directory and os.path.exists(persist_directory):
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        elif pdf_path and os.path.exists(pdf_path):
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
            texts = text_splitter.split_documents(texts)
            self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=persist_directory)
        else:
            self.vectordb = None
            raise ValueError("NO PDF found")
        return self.vectordb
    
    def retreival_qa_chain(self):
    
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k":3})
        hf_llm = HuggingFacePipeline(pipeline=self.llm,model_id=self.config["llm"])
        self.qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",retriever=self.retriever)
        self.qa.combine_documents_chain.verbose = True
        self.qa.return_source_documents = True

    def answer_query(self,question:str) ->str:    
        answer_dict = self.qa({"query":question,})
        return answer_dict
    