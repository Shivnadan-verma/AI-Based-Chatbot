from openai import OpenAI
import os

MONSTER_API_KEY="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImJjZGUxNDkwYzg2MzA4NGEwZWY0MjdkOGEwM2NiZmRlIiwiY3JlYXRlZF9hdCI6IjIwMjQtMDQtMDFUMDg6NTk6MTYuMzY5MTIxIn0.GRUn3Kzp_KnrkRsevNRwzVfpiJqvLJ6CcYAFAla4bJQ"
generation_model_name : str


temperature:float= 0.9
top_p=0.9
max_tokens:int= 2048
stream: bool= True
llm_name:str ="Meta-Llama"



monster_client = OpenAI(
            base_url="https://llm.monsterapi.ai/v1/",
            api_key=str(MONSTER_API_KEY))


monster_ai_model_name = {
                "Google-Gemma": "google/gemma-2-9b-it",
                "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
                "Microsoft-Phi": "microsoft/Phi-3-mini-4k-instruct",
                "Meta-Llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            }

message=[
        {"role": "user", "content": "Tell me about the digital frauds"}]



response  = monster_client.chat.completions.create(
    model=monster_ai_model_name[llm_name],
    messages=message,
    temperature = temperature,
    top_p = top_p,
    max_tokens = max_tokens,
    stream=stream
)


generated_text = ""
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        generated_text += chunk.choices[0].delta.content
        

print(generated_text)
