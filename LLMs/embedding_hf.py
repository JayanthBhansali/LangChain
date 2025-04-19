from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

txt = "I'm learning langchain with opensource"
vector = embeddings.embed_query('txt')

print(str(vector))