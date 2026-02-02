# agents/decision_agent.py

from agents.query_agent import QueryAgent
from agents.sentiment_agent import SentimentAgent
from agents.rag_agent import RAGAgent
from agents.ml_agent import MLAgent
import pandas as pd



class DecisionAgent:
    def __init__(self):
        self.query_agent = QueryAgent()

        self.sentiment_agent = SentimentAgent(
            r"E:/New folder (11)/ai_customer_intelligence/data/Reviews.csv"
        )

        self.rag_agent = RAGAgent(
            r"E:/New folder (11)/ai_customer_intelligence/data/Reviews.csv"
        )

        self.ml_agent = MLAgent(
            r"E:/New folder (11)/ai_customer_intelligence/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )

        self.ml_agent.train()

    def run(self, query: str):
        intent = self.query_agent.understand(query)

        response = {
            "query": query,
            "intent": intent
        }

        # -------- Sentiment --------
        if intent.get("sentiment"):
            response["sentiment"] = self.sentiment_agent.predict(query)

        # -------- RAG --------
        if intent.get("rag"):
            response["evidence"] = self.rag_agent.retrieve(query, top_k=3)

        # -------- Churn Prediction --------
        if intent.get("churn"):
            sample = self.ml_agent.X_test.iloc[[0]]
            response["churn_prediction"] = int(
                self.ml_agent.predict(sample)[0]
            )

        return response
