from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # ✅ More reliable supported model
    task="text-generation",
    provider="hf-inference"  # ✅ Most stable provider
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of France?")
print("LLM Result:", result.content)
