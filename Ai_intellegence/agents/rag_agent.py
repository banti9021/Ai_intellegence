import pandas as pd

csv_path = "E:/New folder (11)/ai_customer_intelligence/data/Reviews.csv"
df = pd.read_csv(csv_path)

print("CSV Loaded Successfully")
print(df.head())
df=df.dropna()
print(df.head())
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
print(df.head())
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Text cleaning function ----------------
def clean_text(Text):
    text = str(Text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# ---------------- RAG Agent Class ----------------
class RAGAgent:
    def __init__(self, csv_path):
        # Load CSV
        self.df = pd.read_csv(csv_path)
        print("CSV Loaded Successfully")
        print(self.df.head())

        # Ensure 'Text' column exists
        if 'Text' not in self.df.columns:
            raise ValueError("CSV must contain 'Text' column!")

        # Drop missing tweets
        self.df = self.df.dropna(subset=['Text'])
        print("After dropping missing tweets:")
        print(self.df.head())

        # Clean tweets
        self.texts = [clean_text(t) for t in self.df['Text']]

        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.texts)

    # ---------------- Retrieve function ----------------
    def retrieve(self, query, top_k=5):
        query_clean = clean_text(query)
        query_vec = self.vectorizer.transform([query_clean])
        scores = cosine_similarity(query_vec, self.vectors)
        top_idx = scores.argsort()[0][-top_k:][::-1]
        return self.df['Text'].iloc[top_idx].tolist()

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    csv_path = "E:/New folder (11)/ai_customer_intelligence/data/Reviews.csv"
 # replace with your file path
    rag = RAGAgent(csv_path)

    query = "Why are customers unhappy with Spotify?"
    results = rag.retrieve(query, top_k=5)

    print("\nTop Relevant Tweets:")
    for tweet in results:
        print("-", tweet)
