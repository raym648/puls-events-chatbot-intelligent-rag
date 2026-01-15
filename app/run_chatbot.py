# puls-events-chatbot-intelligent-rag/app/run_chatbot.py
# ➡ Interface simple pour tester le chatbot RAG

from rag_chain import generate_answer

print("\nAssistant Puls-Events (RAG)")
print("Tape 'exit' pour quitter.\n")

while True:
    question = input("Utilisateur > ")

    if question.lower() == "exit":
        break

    answer = generate_answer(question)
    print("\nAssistant >", answer, "\n")
