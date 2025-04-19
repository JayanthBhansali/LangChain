from langchain_google_genai import ChatGoogleGenerativeAI as Gemini
from dotenv import load_dotenv

load_dotenv()

model = Gemini(model='gemini-1.5-pro', temperature=1.)

res = model.invoke("What is the capital of India?")
print(res.content)