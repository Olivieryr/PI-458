# Social Engineering Simulator (Defensive) - CLI

Projet scolaire : simulateur defensif d'ingenierie sociale pour sensibilisation.
Objectifs V1 :
- CLI interactive
- Style Router (adaptation ton)
- PII Redaction (aucune PII stockee/traitee)
- Policy Guard (anti-derive)
- Scoring + Debrief

## Lancement
1) Installer Ollama et telecharger le modele:
- `ollama pull llama3.2:3b`

2) Installer dependances Python:
- `pip install -r requirements.txt`

3) (Optionnel) Configurer OpenAI:
- creer un fichier `.env` et ajouter: `OPENAI_API_KEY=sk-...`

4) Lancer:
- `python main.py`

## Commandes CLI (resume)
Arguments:
- `--mode local|openai|phi3` : selection du provider (ollama / OpenAI / phi3:mini).
- `--model <nom>` : override du modele pour le mode choisi.
- `--ner` : active la redaction PII via spaCy NER (desactive par defaut).
- `--ner-mode balanced|strict` : mode NER (strict = plus de faux positifs).
- `--ner-model <nom>` : modele spaCy (ex: `fr_core_news_sm`).
- `--profiling` : affiche les timings de chaque etape.

Commandes en session:
- `/help` : aide.
- `/exit` : quitter.
- `/scenario` : lister les scenarios.
- `/scenario <nom>` : changer de scenario.
