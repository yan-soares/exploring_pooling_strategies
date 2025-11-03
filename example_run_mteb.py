import mteb
from sentence_transformers import SentenceTransformer

# Select model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)

# Select tasks
tasks = mteb.get_tasks(tasks=["Banking77Classification"])

# evaluate
results = mteb.evaluate(model, tasks=tasks)
df = results.to_dataframe()
print(df)