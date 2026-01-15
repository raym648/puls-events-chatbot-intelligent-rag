# puls-events-chatbot-intelligent-rag/app/run_chatbot.py
# ➡ Interface simple pour tester le chatbot RAG

from app.rag_service import RAGService

service = RAGService()

print("\nAssistant Puls-Events (RAG)")
print("Tape 'exit' pour quitter.\n")

while True:
    question = input("Utilisateur > ")

    if question.lower() == "exit":
        break

    answer = service.ask(question)
    print("\nAssistant >", answer, "\n")
