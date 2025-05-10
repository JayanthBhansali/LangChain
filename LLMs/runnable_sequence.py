from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)


model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'Generate a GenZ joke on {topic}',
    input_variables = ['topic']
)


prompt2 = PromptTemplate(
    template = 'Explain the joke in a simple way - {text}', 
    input_variables = ['text']
)

parser = StrOutputParser()


chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

res = chain.invoke({'topic':'AI'})
print(res)


