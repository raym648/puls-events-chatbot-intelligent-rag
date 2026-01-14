# Puls-Events – Système RAG pour recommandations culturelles

**Projet-7 - Concevez et déployez un système RAG**

* **✍️ Auteur :** *[Raymond Francius]* 
* **📚 Rôle :** *[Apprenant - Promotion Sept-2025]* — **Engineer AI** — **Openclassrooms**  
* 🗓️ **Date de mise à jour :** *[14-01-2026]*


## Objectif
Ce projet est un POC démontrant la faisabilité d’un **assistant IA** capable de recommander des événements culturels à partir de données OpenAgenda en utilisant une **architecture RAG**.

---

## Stack technique
- LangChain (orchestration RAG)
- FAISS (base vectorielle)
- Mistral AI (LLM)
- HuggingFace (embeddings)
- FastAPI (API REST)

---

## Installation
```bash
git clone <repo>
cd puls-events-chatbot-intelligent-rag
python -m venv env
source env/bin/activate

pip uninstall -y faiss faiss-cpu
pip install -r requirements.txt
python3 -m pip install -r requirements.txt
python -m pip install -U langchain-community

```
---

**Créer le fichier .env :**
```bash
MISTRAL_API_KEY=your_key_here
```
**Vérification**
```bash
python scripts/test_environment.py
```

**Vous devez voir :**
```bash
Python OK
FAISS OK
LangChain FAISS OK
HuggingFace Embeddings OK
Mistral Client OK
Tous les composants sont correctement installés
```
**Structure**
```bash
app/         API FastAPI
data/        Données OpenAgenda
scripts/     Scripts de traitement
vectorstore/ Index FAISS
tests/       Tests unitaires
```
---

# 📊 Résumé exécutif Étape 1
```bash
| Élément | Statut |
|-------|------|
| Environnement isolé | ✅ |
| Librairies compatibles | ✅ |
| FAISS CPU | ✅ |
| Mistral sécurisé | ✅ |
| Projet clonable | ✅ |
| Test automatisé | ✅ |
```
---