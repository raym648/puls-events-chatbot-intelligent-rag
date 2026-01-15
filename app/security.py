# puls-events-chatbot-intelligent-rag/app/security.py
# 👉 Protection des endpoints sensibles

import os
from fastapi import HTTPException, Header


def verify_admin_token(x_admin_token: str = Header(None)):
    """
    Vérifie le token d'administration pour protéger /rebuild.

    - Si ADMIN_TOKEN n'est pas défini → mode développement → accès autorisé
    - Si ADMIN_TOKEN est défini → vérification stricte
    """
    expected_token = os.getenv("ADMIN_TOKEN")

    # Mode développement : pas de token configuré
    if expected_token is None:
        return

    # Mode sécurisé
    if x_admin_token != expected_token:
        raise HTTPException(status_code=403, detail="Unauthorized operation")
