import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap

# --------------------------------------------------------------
# Load the HuggingFaceHub API token from the .env file
# --------------------------------------------------------------

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# --------------------------------------------------------------
# Load the LLM model from the HuggingFaceHub
# --------------------------------------------------------------

repo_id = "tiiuae/falcon-40b"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.8, "max_new_tokens": 500}
)


# --------------------------------------------------------------
# Create a PromptTemplate and LLMChain
# --------------------------------------------------------------
template = """Question: {question}

Context: addon is sub product of base"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)


# --------------------------------------------------------------
# Run the LLMChain
# --------------------------------------------------------------

question = "what is addon?"
response = llm_chain.run(question)
wrapped_text = textwrap.fill(
    response, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)



