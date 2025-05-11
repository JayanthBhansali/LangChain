from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)


model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'Generate a funny GenZ joke on {topic}',
    input_variables = ['topic']
)


prompt2 = PromptTemplate(
    template = 'Explain the joke in a simple way - {text}', 
    input_variables = ['text']
)

parser = StrOutputParser()

# the issue was that thejoke wasn't printing
# we can use the RunnablePassthrough to get the joke
# in parrallel we can get its explanation

# chain1 to generate the joke
chain1 = RunnableSequence(prompt, model, parser)

# chain2 to keep the joke and get the explanation
chain2 = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
})

# chain3 to connect the two chains
# using declarative syntax instead of RunnableSequence
# chain3 = RunnableSequence(chain1, chain2)
chain3 = chain1 | chain2

res = chain3.invoke({'topic':'ML'})
print(res)


