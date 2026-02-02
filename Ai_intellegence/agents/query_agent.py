class QueryAgent:
    def understand(self, query: str) -> dict:
        query = query.lower()

        intent = {
            "sentiment": False,
            "churn": False,
            "rag": False,
            "action": False
        }

        # RAG related questions
        if any(word in query for word in ["why", "who", "what", "which", "best"]):
            intent["rag"] = True

        # Sentiment related
        if any(word in query for word in ["disappointed", "good", "negative", "positive", "very", "best"]):
            intent["sentiment"] = True

        # Churn related
        if any(word in query for word in ["leave", "cancel", "churn", "stop"]):
            intent["churn"] = True

        # Action / recommendation
        if any(word in query for word in ["improve", "predict", "suggest", "recommend"]):
            intent["action"] = True

        return intent
