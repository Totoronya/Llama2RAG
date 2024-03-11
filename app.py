# Import streamlit for app dev
import streamlit as st

# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
# Import torch for datatype attributes
import torch
# Import the prompt wrapper...but for llama index
from llama_index.core import PromptTemplate
# Import the llama index HF Wrapper
from llama_index.llms.huggingface import HuggingFaceLLM
# Bring in embeddings wrapper
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# Bring in stuff to change service context
from llama_index.core import set_global_service_context
from llama_index.core import ServiceContext
# Import deps to load documents
from llama_index.core import VectorStoreIndex, download_loader
from pathlib import Path

# Define variable to hold llama2 weights naming
name = "meta-llama/Llama-2-7b-chat-hf"
# Set auth token variable from hugging face
auth_token = "mytoken"


quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_threshold=200.0)
@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='../data/llama/llama-2-7b/', token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='../data/llama/llama-2-7b/'
                            , token=auth_token, torch_dtype=torch.float16,
                            rope_scaling={"type": "dynamic", "factor": 2})

    return tokenizer, model
tokenizer, model = get_tokenizer_model()

# Create a system prompt
base_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, >
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question><</SYS>>

There's a llama in my garden 😱 What should I do? [/INST]

"""
# Throw together the query wrapper
query_wrapper_prompt = PromptTemplate("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=base_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Create and dl embeddings instance
model_kwargs = {'trust_remote_code': True, "load_in_8bit":True}
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)

# Download PDF Loader
PyMuPDFReader = download_loader("PyMuPDFReader")
# Create PDF Loader
loader = PyMuPDFReader()
# Load documents
documents = loader.load(file_path=Path('./data/annualreport.pdf'), metadata=True)

# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)
# Setup index query engine using LLM
query_engine = index.as_query_engine()

# Create centered main title
st.title('  Llama Banker')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    response = query_engine.query(prompt)
    # ...and write it out to the screen
    print("doing response")
    st.write(response)

    # Display raw response object
    with st.expander('Response Object'):
        st.write(response)
    # Display source text
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())
