# Puls-Events – Système RAG pour recommandations culturelles

**Projet-7 - Concevez et déployez un système RAG**

* **✍️ Auteur :** *[Raymond Francius]* 
* **📚 Rôle :** *[Apprenant - Promotion Sept-2025]* — **Engineer AI** — **Openclassrooms**  
* 🗓️ **Date de mise à jour :** *[14-01-2026]*


## 🎯 Objectif
Ce projet est un **POC (Proof of Concept)** démontrant la faisabilité d’un **assistant IA** capable de recommander des événements culturels à partir des données **OpenAgenda**, en utilisant une **architecture RAG (Retrieval-Augmented Generation)**.

L’objectif métier est de permettre aux équipes **produit** et **marketing** de Puls-Events de tester un chatbot capable de :
- Comprendre des questions utilisateurs,
- Rechercher les événements pertinents,
- Générer des réponses naturelles, fiables et contextualisées.

---

## 🧠 Architecture globale (RAG)
```bash
Utilisateur
│
▼
FastAPI ──▶ LangChain ──▶ FAISS ──▶ OpenAgenda Events
│ ▲
▼ │
Mistral LLM ◀── Contextes vectorisés (embeddings)
```

**Le système repose sur :**
- Une **base vectorielle FAISS** contenant les descriptions d’événements,
- Un **LLM Mistral** pour la génération de réponses,
- **LangChain** pour orchestrer la recherche + génération,
- Une **API FastAPI** exposant le chatbot.

---

## 🧩 Stack technique
```bash
| Composant               | Rôle                             |
|-------------------------|----------------------------------|
| **LangChain**           | Orchestration RAG                |
| **FAISS**               | Recherche sémantique vectorielle |
| **Mistral AI**          | Modèle de langage (LLM)          |
| **HuggingFace**         | Embeddings                       |
| **FastAPI**             | API REST                         |
| **RAGAS**               | Évaluation qualité RAG           |
| **GitHub Actions**      | CI/CD                            |
| **Docker**              | Déploiement                      |
| **Hugging Face Spaces** | Hébergement                      |
```
---

## ⚙️ Installation

```bash
git clone <repo>
cd puls-events-chatbot-intelligent-rag

python -m venv env
source env/bin/activate

pip uninstall -y faiss faiss-cpu
pip install -r requirements.txt
python -m pip install -U langchain-community
```

## 🔐 Variables d’environnement
**Créer le fichier .env :**
```bash
MISTRAL_API_KEY=your_key_here
ADMIN_TOKEN=secure_admin_token
⚠️ Ce fichier est ignoré par Git pour des raisons de sécurité.
```

## 🧪 Vérification de l’environnement
```bash
python scripts/test_environment.py
```
**Sortie attendue :**
```bash
Python OK
FAISS OK
LangChain FAISS OK
HuggingFace Embeddings OK
Mistral Client OK
Tous les composants sont correctement installés
```

## 📂 Structure du projet
```bash
puls-events-chatbot-intelligent-rag/
│
├── app/          API FastAPI + RAG
├── data/         Données OpenAgenda
├── scripts/      Préprocessing & indexation
├── vectorstore/  Index FAISS
├── tests/        Tests unitaires & RAGAS
├── .env
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 Étape 1 : Environnement
```bash
| Élément                | Statut |
|------------------------|--------|
| Environnement isolé    |   ✅   |
| Librairies compatibles |   ✅   |
| FAISS CPU              |   ✅   |
| Mistral sécurisé       |   ✅   |
| Projet clonable        |   ✅   |
| Test automatisé        |   ✅   |
```

---

## 📊 Étape 2 – Données OpenAgenda
**Pipeline :**
- Récupération via API OpenAgenda
- Filtrage géographique & temporel
- Nettoyage des champs manquants
- Création de textes exploitables
- Génération d’embeddings

**Fichiers produits :**
```bash
data/raw_events.json
data/cleaned_events.csv
data/cleaned_events_with_embeddings.pkl
```

---

## 🧠 Étape 3 – Base vectorielle FAISS
**Chaque événement est stocké avec :**
- Son embedding
- Son titre
- Sa ville
- Sa date
- Son URL

**Fichiers :**
```bash
vectorstore/faiss.index
vectorstore/faiss_store.pkl
```

**Tests :**
```bash
python scripts/test_faiss_search.py
```

---

## 🤖 Étape 4 – Système RAG
**Le moteur RAG :**
- Récupère les événements les plus proches sémantiquement
- Injecte leur contenu dans le prompt
- Génère une réponse Mistral contextualisée

**Test :**
```bash
pytest tests/test_rag.py
```

---

## 🌐 Étape 5 – API FastAPI
**Démarrage :**
```bash
uvicorn app.main:app --reload
```

**Swagger :**
```bash
http://localhost:8000/docs
```

**Endpoints :**
```bash
Route	Rôle
POST /ask	Poser une question
POST /rebuild	Recalculer l’index FAISS
```

**Test :**
```bash
python scripts/api_test.py
```

---

## 📈 Évaluation automatique (RAGAS)
```bash
python tests/evaluate_rag.py
```

**Mesures :**
- Context Precision
- Answer Faithfulness
- Answer Relevance

---

### 🚀 Déploiement
**Le projet est déployé automatiquement via GitHub Actions vers :**
- Hugging Face Space API
- Hugging Face Space Dashboard

**À chaque git push main, le pipeline :**
- Exécute tous les tests
- Évalue la qualité RAG
- Construit les images Docker
- Déploie en production

### 🏁 Résultat
**Puls-Events dispose maintenant :**
- D’un chatbot IA opérationnel
- D’une API REST sécurisée
- D’un dashboard
- D’un pipeline MLOps complet
- D’une base vectorielle sémantique

---

## 🧠🎭 15 exemples de requêtes sur les événements culturels
**🎷 Musique / Jazz** 
- “Quels concerts de jazz sont prévus à Paris en mars 2025 ?”  
- “Je cherche un concert de jazz manouche à Paris fin mars.”  
- “Y a-t-il un concert de jazz le 15 mars à Paris ?”  
- “Quels événements musicaux ont lieu à la Bellevilloise ce mois-ci ?”  

**🎭 Théâtre & danse** 
- “Quelles pièces de théâtre classiques sont jouées à Paris en mars ?”  
- “Je voudrais voir un spectacle de danse contemporaine à Paris.”  
- “Y a-t-il des spectacles d’improvisation théâtrale ce mois-ci ?”  

**🖼️ Expositions** 
- “Quelles expositions sont visibles à Paris au printemps 2025 ?”  
- “Existe-t-il une exposition de photographie à Paris en mars ?”  
- “Je cherche une exposition d’art moderne à Paris.”  

**🎬 Cinéma & conférences** 
- “Y a-t-il des événements autour du cinéma en mars 2025 à Paris ?”  
- “Des conférences qui parlent d’intelligence artificielle et de culture ?”  

**👨‍👩‍👧 Famille & médiation culturelle** 
- “Quels événements culturels sont adaptés aux enfants à Paris ?”  
- “Je cherche une activité culturelle pour un dimanche en mars.”  
- “Y a-t-il des visites guidées culturelles prévues fin mars à Paris ?”  

---
### 🌐 Démo & Accès API **Dashboard Hugging Face** :  
[https://remdev-ai-rag-dashboard.hf.space](https://remdev-ai-rag-dashboard.hf.space)

---
***Ce POC démontre la faisabilité industrielle d’un assistant de recommandation culturelle basé sur RAG.***
