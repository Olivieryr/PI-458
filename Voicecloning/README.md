# 🛡️ Social Engineering Simulator

Simulateur CLI de sensibilisation à l'ingénierie sociale. Un agent LLM joue le rôle d'un attaquant (auditeur interne fictif) et tente d'obtenir des informations sensibles de l'utilisateur, qui doit apprendre à résister.

---

## 📋 Prérequis

- Python **3.10+**
- [Ollama](https://ollama.com/) installé et en cours d'exécution en local
- GPU NVIDIA recommandé (CUDA) pour la synthèse vocale XTTS v2 en temps réel
- Un fichier audio de référence `voix_norm2.wav` à la racine du projet (voix de clonage TTS)

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd social-engineering-simulator

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le modèle spaCy (optionnel, pour la détection NER)
python -m spacy download fr_core_news_sm

# 5. Configurer l'environnement
cp .env.example .env
# Éditer .env selon vos besoins
```

---

## 🚀 Lancement

```bash
python main.py
```

### Options disponibles

| Argument | Description | Exemple |
|---|---|---|
| `--text-model` | Surcharge le modèle de génération de texte | `--text-model mistral` |
| `--transition-model` | Surcharge le modèle de classification des intentions | `--transition-model llama3` |
| `--ner` | Active la détection NER via spaCy | `--ner` |
| `--ner-mode` | Niveau de strictesse NER : `balanced` ou `strict` | `--ner-mode strict` |
| `--ner-model` | Modèle spaCy à utiliser | `--ner-model fr_core_news_lg` |
| `--profiling` | Affiche les temps d'exécution par étape | `--profiling` |

### Exemple complet

```bash
python main.py --text-model mistral --ner --ner-mode strict --profiling
```

---

## 🗂️ Structure du projet

```
.
├── main.py                  # Point d'entrée principal
├── config.py                # Chargement de la configuration
├── voix_norm2.wav           # Fichier de référence pour la voix TTS
├── requirements.txt
├── .env
└── core/
    ├── dialog_manager.py    # Gestion des états et du scénario
    ├── llm_client.py        # Client Ollama
    ├── pii_detector.py      # Détection et masquage regex des PII
    ├── pii_ner.py           # Détection NER des PII (spaCy)
    ├── policy_guard.py      # Post-vérification des sorties LLM
    ├── safe_log.py          # Logger avec masquage des données sensibles
    ├── style_router.py      # Détection et adaptation du style conversationnel
    └── transition_engine.py # Classification des intentions utilisateur
```

---

## 🎮 Commandes en session

| Commande | Action |
|---|---|
| `/exit` | Quitter la session |
| `/help` | Afficher l'aide |
| `/scenario` | Lister les scénarios disponibles |
| `/scenario <nom>` | Changer de scénario à la volée |

---

## 🔒 Pipeline de sécurité

Chaque message utilisateur passe par les étapes suivantes avant d'atteindre le LLM :

1. **Redaction PII** — masquage regex des données personnelles (noms, emails, téléphones…)
2. **NER optionnel** — détection des entités nommées via spaCy
3. **Détection de style** — adaptation du ton de l'agent (formel, informel, verbeux…)
4. **Classification d'intention** — moteur de transition pour faire évoluer le scénario
5. **Post-check** — vérification de la sortie LLM avant affichage et log

---

## 🔊 Synthèse vocale (TTS)

Le simulateur utilise [Coqui XTTS v2](https://github.com/coqui-ai/TTS) pour la synthèse vocale en temps réel avec clonage de voix. La lecture audio se fait dans un thread dédié pour ne pas bloquer la génération de texte en streaming.

- Le modèle est téléchargé automatiquement au premier lancement (~2 Go)
- Un GPU CUDA accélère significativement la synthèse
- Le fichier `voix_norm2.wav` sert de référence pour le clonage de voix

---

## ⚠️ Avertissement

Ce simulateur est conçu **exclusivement à des fins pédagogiques** de sensibilisation à la sécurité. Il ne doit pas être utilisé pour mener de véritables attaques d'ingénierie sociale.
