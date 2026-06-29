import httpx
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("model")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

if not MODEL:
    raise ValueError("MODEL environment variable not set.")

client = Groq(api_key=GROQ_API_KEY, http_client=httpx.Client(verify=False))


def ask_llm(prompt):
    try:
        response = client.chat.completions.create(model=MODEL,   #type: ignore
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
