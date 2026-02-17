import os
import sys
import shutil
import glob
import subprocess
import configparser

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = "/app"
INPUT_DIR = "/app/inputs"
OUTPUT_DIR = "/app/outputs"
VIDEOFAKE_DIR = "/app/VideoFake"
EASY_DIR = "/app/Easy-Wav2Lip"

# On s'assure que les dossiers existent
os.makedirs(VIDEOFAKE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chemins des fichiers sources
MY_FACE = os.path.join(INPUT_DIR, "roman.jpeg")
MY_BODY_VIDEO = os.path.join(INPUT_DIR, "floVF.mp4")
MY_AUDIO = os.path.join(INPUT_DIR, "audio.wav")

TEMP_SWAP = os.path.join(VIDEOFAKE_DIR, "temp_swapped.mp4")
FINAL_RESULT = os.path.join(OUTPUT_DIR, "RESULTAT_FINAL.mp4")

def run_command(command):
    """Execution des commandes shell"""
    print(f"\n> Execution de : {command}")
    subprocess.run(command, shell=True, check=True)

print("--- DEBUT DE LA PIPELINE IA ---")

# --- ETAPE 1 : FACESWAP (ROOP) ---
print("\n[1/2] SECTION FACESWAP (ROOP)")

if os.path.exists(TEMP_SWAP):
    print("Fichier FaceSwap deja present, on passe a la suite.")
else:
    if not os.path.exists(f"{VIDEOFAKE_DIR}/roop"):
        print("Clonage du repo FaceSwap...")
        run_command(f"git clone https://github.com/neuromodern/VideoFake.git {VIDEOFAKE_DIR}")

    os.chdir(f"{VIDEOFAKE_DIR}/roop")

    print("Installation des dependances de derniere minute...")
    # On installe TOUT ce qui manquait
    run_command("pip install customtkinter onnxruntime tkinterdnd2 gdown ipython moviepy librosa scipy")

    cmd_swap = (
        f"python run.py --execution-provider cpu "
        f"--source {MY_FACE} -t {MY_BODY_VIDEO} -o {TEMP_SWAP} "
        f"--frame-processor face_swapper --output-video-encode libx264 "
        f"--output-video-quality 18 --keep-fps"
    )
    
    try:
        # On force le mode sans fenetre
        run_command(f"export TK_SILENCE_DEPRECATION=1 && {cmd_swap}")
    except Exception as e:
        print(f"Erreur durant le FaceSwap : {e}")
        sys.exit(1)

if not os.path.exists(TEMP_SWAP):
    print("Erreur : Le fichier FaceSwap n'a pas ete genere.")
    sys.exit(1)

# --- FIX RADICAL COMPATIBILITÉ TORCHVISION ---
print("\n[FIX] Correction physique des fichiers basicsr...")
paths = glob.glob("/usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py")
if paths:
    file_path = paths[0]
    with open(file_path, 'r') as f:
        content = f.read()
    
    old_line = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    new_line = "from torchvision.transforms.functional import rgb_to_grayscale"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(file_path, 'w') as f:
            f.write(content)
        print(" Patch appliqué.")
    else:
        print(" Fichier déjà patché.")
else:
    print(" Attention : basicsr non trouvé (peut-etre normal).")

# --- ETAPE 2 : WAV2LIP (LIP SYNC) ---
print("\n[2/2] SECTION WAV2LIP (SYNCHRO LABIALE)")

if not os.path.exists(EASY_DIR):
    print("Clonage du repo Easy-Wav2Lip...")
    run_command(f"git clone -b v8.3 https://github.com/anothermartz/Easy-Wav2Lip.git {EASY_DIR}")

os.chdir(EASY_DIR)
os.makedirs("face_alignment", exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Telechargement des modeles
models = {
    "checkpoints/predictor.pkl": "https://huggingface.co/spaces/kaushikpandav/Lipsync_Antriksh_AI/resolve/main/checkpoints/predictor.pkl",
    "checkpoints/mouth_detector.pkl": "https://huggingface.co/spaces/kaushikpandav/Lipsync_Antriksh_AI/resolve/main/checkpoints/mouth_detector.pkl",
    "checkpoints/Wav2Lip.pth": "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth"
}

for path, url in models.items():
    if not os.path.exists(path):
        print(f"Telechargement : {path}")
        run_command(f"wget -q -O {path} {url}")

# CONFIGURATION WAV2LIP
config = configparser.ConfigParser()
config['OPTIONS'] = {
    'video_file': TEMP_SWAP,
    'vocal_file': MY_AUDIO,
    'quality': "Improved",
    'output_height': "full resolution",
    'wav2lip_version': "Wav2Lip",
    'use_previous_tracking_data': "True",
    'nosmooth': "True",
    'preview_window': "False" 
}
config['PADDING'] = {'U': 0, 'D': 0, 'L': 0, 'R': 0}
config['MASK'] = {'size': 1.5, 'feathering': 1, 'mouth_tracking': "False", 'debug_mask': "False"}
config['OTHER'] = {'batch_process': "False", 'output_suffix': "_FINAL", 'include_settings_in_suffix': "False", 'preview_input': "False", 'preview_settings': "False", 'frame_to_preview': 100}

with open('config.ini', 'w') as f:
    config.write(f)

print("Lancement du Lip-Sync...")
try:
    run_command("python run.py")
except Exception as e:
    # On continue meme si erreur code 1, car le fichier peut exister
    print(f"Fin du script (Code retour: {e})")

# --- ETAPE 3 : RECUPERATION ET COPIE FINALE ---
print("\n--- RECUPERATION DU RESULTAT ---")

# Chemin EXACT basé sur tes logs précédents
source_exacte = "/app/VideoFake/temp_swapped_audio_FINAL.mp4"

if os.path.exists(source_exacte):
    print(f" Fichier trouvé à la source : {source_exacte}")
    try:
        shutil.copy(source_exacte, FINAL_RESULT)
        print("---------------------------------------------------")
        print(f" Video prête sur le bureau:")
        print(f" outputs/RESULTAT_FINAL.mp4")
        print("---------------------------------------------------")
    except Exception as e:
        print(f" Erreur de copie vers le bureau : {e}")
else:
    print(f" Fichier exact non trouvé ({source_exacte}). Recherche large...")
    # Plan de secours : on cherche n'importe quel MP4 récent dans VideoFake
    fichiers_mp4 = glob.glob("/app/VideoFake/*.mp4")
    if fichiers_mp4:
        # On exclut le fichier temporaire temp_swapped
        candidats = [f for f in fichiers_mp4 if "temp_swapped.mp4" not in f]
        if candidats:
            le_plus_recent = max(candidats, key=os.path.getctime)
            shutil.copy(le_plus_recent, FINAL_RESULT)
            print(f" Vidéo trouvée et copiée : {FINAL_RESULT}")
        else:
            print(" Aucun fichier final détecté.")
    else:
        print(" Dossier VideoFake vide ou fichier absent.")