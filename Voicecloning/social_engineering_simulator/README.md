# Social Engineering Simulator (Defensive) - CLI

Projet scolaire : simulateur defensif d'ingenierie sociale pour sensibilisation.
Objectifs V1 :
- CLI interactive
- Dialog Manager hybride (FSM + intents)
- Transition Engine (phi3:mini) pour l'intention
- Text Generator (llama3.2:3b) pour la reponse
- Style Router (adaptation ton)
- PII Redaction (aucune PII stockee/traitee)
- Policy Guard (anti-derive)
- Scoring + Debrief

## Lancement
1) Installer Ollama et telecharger le modele:
- `ollama pull llama3.2:3b`
- `ollama pull phi3:mini`

2) Installer dependances Python:
- `pip install -r requirements.txt`

3) Lancer:
- `python main.py`

## Commandes CLI (resume)
Arguments:
- `--text-model <nom>` : override du modele de generation (llama3.2:3b).
- `--transition-model <nom>` : override du modele de transition (phi3:mini).
- `--ner` : active la redaction PII via spaCy NER (desactive par defaut).
- `--ner-mode balanced|strict` : mode NER (strict = plus de faux positifs).
- `--ner-model <nom>` : modele spaCy (ex: `fr_core_news_sm`).
- `--profiling` : affiche les timings de chaque etape.

Commandes en session:
- `/help` : aide.
- `/exit` : quitter.
- `/scenario` : lister les scenarios.
- `/scenario <nom>` : changer de scenario.
