from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch

# Device setup
device = 0 if torch.cuda.is_available() else -1
print("Device set to:", "cuda" if device == 0 else "cpu")

# HuggingFace model
pipe = pipeline("text-generation", 
                model="tiiuae/falcon-7b-instruct", 
                tokenizer="tiiuae/falcon-7b-instruct", 
                max_new_tokens=256, 
                do_sample=True,
                temperature=0.7,
                device=device)

llm = HuggingFacePipeline(pipeline=pipe)
chat_model = RunnableLambda(lambda messages: AIMessage(content=llm.invoke(messages[-1].content)))

# Chat History
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain.")
]

result = chat_model.invoke(messages)
messages.append(result)

# Print updated conversation
for msg in messages:
    role = msg.__class__.__name__.replace("Message", "")
    print(f"{role}: {msg.content}")
