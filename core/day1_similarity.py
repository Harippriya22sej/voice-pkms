from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Remind me to call Ravi about the demo.",
    "I need to phone Ravi regarding the project presentation.",
    "Buy vegetables from the market tomorrow.",
]

embeddings = model.encode(sentences)

print("1 vs 2:", util.cos_sim(embeddings[0], embeddings[1]).item())
print("1 vs 3:", util.cos_sim(embeddings[0], embeddings[2]).item())