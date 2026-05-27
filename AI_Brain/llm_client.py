from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = os.getenv("model")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

if not model:
    raise ValueError("model environment variable not set.")

client = Groq(api_key=GROQ_API_KEY)


def ask_llm(prompt):
    try:
        response = client.chat.completions.create(model=model,
                                                  messages=[{"role": "system",
                                                            "content": ("You are an expert Machine Learning Engineer and Data Scientist.")},
                                                            {"role": "user","content": prompt}],
                                                  temperature=0.3, 
                                                  max_tokens=1024)

        output = (response.choices[0].message.content)
        
        return output

    except Exception as e:
        return (f"LLM Error: {str(e)}")