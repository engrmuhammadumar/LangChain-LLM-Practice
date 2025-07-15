from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# HuggingFace text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)

# Wrap it with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

print("Chatbot is ready! Type 'exit' to quit.\n")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = llm.invoke(user_input)
    print("AI:", response)
