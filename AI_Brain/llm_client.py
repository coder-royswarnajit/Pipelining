import httpx
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(r"C:/Users/309168/Desktop/CODES/Pipelining (1)/AI_Brain/.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = os.getenv("model")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

if not model:
    raise ValueError("model environment variable not set.")

client = Groq(api_key=GROQ_API_KEY, http_client=httpx.Client(verify=False))


def ask_llm(prompt):
    try:
        response = client.chat.completions.create(model=model,
                                                  messages=[{"role": "system",
                                                            "content": ("You are an expert Machine Learning Engineer and Data Scientist.")},
                                                            {"role": "user","content": prompt}],
                                                  temperature=0.3, 
                                                  max_tokens=4096)

        output = (response.choices[0].message.content)
        
        return output

    except Exception as e:
        print("LLM ERROR:")
        print(type(e))
        print(e)
        raise

'''

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
model = os.getenv("model", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com" 
)

def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Machine Learning Engineer and Data Scientist."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )

        output = response.choices[0].message.content
        return output

    except Exception as e:
        return f"LLM Error: {str(e)}"'''