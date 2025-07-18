from transformers import pipeline

# Use Hugging Face's fill-mask pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Let's test a masked sentence
result = fill_mask("The cat sat on the [MASK].")

# Print predictions
for r in result:
    print(f"{r['sequence']}  -->  score: {r['score']:.4f}")
