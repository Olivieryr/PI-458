import requests
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import json
import os
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
from collections import Counter

class CloudPhotoOSINT:
    def __init__(self):
        print("⏳ Chargement des modèles...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        print("✅ Modèles chargés ! (YOLO + BLIP + Interprétation OSINT)")

    def local_vision(self, image):
        """Vision IA locale avec BLIP"""
        inputs = self.processor(image, return_tensors="pt")
        out = self.blip_model.generate(**inputs, max_length=100)
        description = self.processor.decode(out[0], skip_special_tokens=True)
        return description

    def analyze(self, image_path_or_url):
        """Analyse une seule image"""
        temp_file = None
        if image_path_or_url.startswith('http'):
            resp = requests.get(image_path_or_url)
            img = Image.open(BytesIO(resp.content))
            import uuid
            temp_file = f'temp_{uuid.uuid4().hex[:8]}.jpg'
            img.save(temp_file)
            local_path = temp_file
        else:
            local_path = image_path_or_url
            img = Image.open(local_path)
        
        try:
            results = self.yolo_model(img)
            objects = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        if conf > 0.5:
                            objects.append(r.names[cls])
            
            objects_str = ", ".join(set(objects)) if objects else "Aucun"
            vision_result = self.local_vision(img)
            
            rapport = {
                'image': str(image_path_or_url),
                'objets_detectes': list(set(objects)),
                'description_ia': vision_result,
                'nombre_objets': len(set(objects))
            }
            
            return rapport
            
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

    def interpret_profile(self, all_reports):
        """🧠 INTERPRÉTATION OSINT : Déduit le profil de la personne"""
        
        # Collecte tous les objets
        all_objects = []
        descriptions = []
        for r in all_reports:
            all_objects.extend(r['objets_detectes'])
            descriptions.append(r['description_ia'].lower())
        
        object_counts = Counter(all_objects)
        desc_text = " ".join(descriptions)
        
        interpretation = {
            'animaux': [],
            'loisirs': [],
            'lifestyle': [],
            'mobilite': [],
            'technologie': [],
            'localisation_probable': [],
            'profil_socio': []
        }
        
        # 🐾 ANIMAUX
        if 'dog' in object_counts:
            freq = object_counts['dog']
            if freq >= 3:
                interpretation['animaux'].append(f"🐕 PROPRIÉTAIRE DE CHIEN (détecté {freq}x) - Présence récurrente, probable propriétaire")
            elif freq == 2:
                interpretation['animaux'].append(f"🐕 Chien détecté {freq}x - Propriétaire possible ou garde régulière")
            else:
                interpretation['animaux'].append("🐕 Chien détecté 1x - Contact ponctuel (ami, famille, parc)")
        
        if 'cat' in object_counts:
            freq = object_counts['cat']
            if freq >= 2:
                interpretation['animaux'].append(f"🐈 PROPRIÉTAIRE DE CHAT (détecté {freq}x)")
            else:
                interpretation['animaux'].append("🐈 Chat détecté - Visite chez des proches ou café à chats")
        
        if 'bird' in object_counts:
            interpretation['animaux'].append("🐦 Intérêt pour les oiseaux - Passionné d'ornithologie ou nature")
        
        if 'horse' in object_counts:
            interpretation['animaux'].append("🐴 Équitation ou visite de centre équestre")
        
        # 🏃 LOISIRS & ACTIVITÉS
        sports = ['sports ball', 'baseball bat', 'tennis racket', 'skateboard', 'surfboard', 'skis', 'snowboard']
        detected_sports = [s for s in sports if s in object_counts]
        if detected_sports:
            interpretation['loisirs'].append(f"🏃 SPORTIF - Pratique: {', '.join(detected_sports)}")
        
        if 'bicycle' in object_counts:
            interpretation['loisirs'].append("🚴 Cycliste - Déplacements vélo ou VTT")
        
        if 'book' in object_counts:
            freq = object_counts['book']
            if freq >= 2:
                interpretation['loisirs'].append(f"📚 LECTEUR ASSIDU ({freq} livres visibles)")
            else:
                interpretation['loisirs'].append("📚 Lecture occasionnelle")
        
        if any(word in desc_text for word in ['beach', 'ocean', 'sea', 'swimming']):
            interpretation['loisirs'].append("🏖️ Activités balnéaires - Vacances en bord de mer")
        
        if any(word in desc_text for word in ['mountain', 'hiking', 'trail']):
            interpretation['loisirs'].append("⛰️ Randonneur - Activités en montagne")
        
        # 🍽️ LIFESTYLE
        if 'wine glass' in object_counts or 'bottle' in object_counts:
            interpretation['lifestyle'].append("🍷 Vie sociale - Sorties/repas entre amis")
        
        if 'dining table' in object_counts:
            freq = object_counts['dining table']
            if freq >= 3:
                interpretation['lifestyle'].append("🍽️ Convivialité importante - Nombreux repas photographiés")
        
        if 'couch' in object_counts or 'bed' in object_counts:
            interpretation['lifestyle'].append("🏠 Photos d'intérieur - Moments à la maison")
        
        if any(word in desc_text for word in ['restaurant', 'cafe', 'bar']):
            interpretation['lifestyle'].append("🍴 Sorties fréquentes au restaurant/café")
        
        # 🚗 MOBILITÉ
        if 'car' in object_counts:
            freq = object_counts['car']
            if freq >= 3:
                interpretation['mobilite'].append(f"🚗 AUTOMOBILISTE ({freq} photos avec voiture)")
            else:
                interpretation['mobilite'].append("🚗 Accès voiture occasionnel")
        
        if 'motorcycle' in object_counts:
            interpretation['mobilite'].append("🏍️ Motard - Passion moto")
        
        if 'airplane' in object_counts or any(word in desc_text for word in ['airport', 'plane', 'flight']):
            interpretation['mobilite'].append("✈️ Voyageur fréquent - Déplacements aériens")
        
        if 'train' in object_counts or 'bus' in object_counts:
            interpretation['mobilite'].append("🚆 Transports en commun réguliers")
        
        # 💻 TECHNOLOGIE
        tech_items = ['laptop', 'cell phone', 'keyboard', 'mouse', 'tv', 'remote']
        detected_tech = [t for t in tech_items if t in object_counts]
        if len(detected_tech) >= 3:
            interpretation['technologie'].append(f"💻 TECH-SAVVY - Équipements: {', '.join(detected_tech)}")
        elif detected_tech:
            interpretation['technologie'].append(f"📱 Utilisateur tech - {', '.join(detected_tech)}")
        
        # 📍 LOCALISATION PROBABLE
        if 'traffic light' in object_counts or 'parking meter' in object_counts:
            interpretation['localisation_probable'].append("🏙️ Zone urbaine - Environnement citadin")
        
        if any(word in desc_text for word in ['eiffel', 'paris', 'louvre']):
            interpretation['localisation_probable'].append("🇫🇷 Paris - Monuments identifiés")
        
        if any(word in desc_text for word in ['snow', 'skiing', 'mountain']):
            interpretation['localisation_probable'].append("⛷️ Région montagneuse/stations de ski")
        
        if any(word in desc_text for word in ['palm tree', 'tropical', 'beach']):
            interpretation['localisation_probable'].append("🌴 Climat tropical/méditerranéen")
        
        # 👤 PROFIL SOCIO-DÉMOGRAPHIQUE
        if 'person' in object_counts:
            freq = object_counts['person']
            if freq >= len(all_reports) * 0.7:
                interpretation['profil_socio'].append(f"👥 TRÈS SOCIAL - Personne(s) dans {freq}/{len(all_reports)} photos")
            elif freq >= 3:
                interpretation['profil_socio'].append(f"👤 Vie sociale active - {freq} photos avec personnes")
        
        if 'tie' in object_counts or 'suitcase' in object_counts:
            interpretation['profil_socio'].append("💼 Professionnel - Contexte business/formel")
        
        if 'backpack' in object_counts:
            interpretation['profil_socio'].append("🎒 Style décontracté/outdoor - Sac à dos fréquent")
        
        if 'handbag' in object_counts:
            interpretation['profil_socio'].append("👜 Attention aux accessoires - Présence de sac à main")
        
        # Filtre les catégories vides
        interpretation = {k: v for k, v in interpretation.items() if v}
        
        return interpretation

    def analyze_folder(self, folder_path, output_dir='rapports_osint'):
        """Analyse TOUT un dossier avec interprétation OSINT"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Dossier introuvable: {folder_path}")
            return
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = []
        for ext in extensions:
            images.extend(folder.glob(f'*{ext}'))
            images.extend(folder.glob(f'*{ext.upper()}'))
        
        if not images:
            print(f"❌ Aucune image trouvée dans {folder_path}")
            return
        
        print(f"\n🗂️  Trouvé {len(images)} image(s) à analyser")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_reports = []
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {img_path.name}...", end=" ")
            try:
                rapport = self.analyze(str(img_path))
                all_reports.append(rapport)
                print("✅")
            except Exception as e:
                print(f"❌ {e}")
                continue
        
        # 🧠 INTERPRÉTATION
        print("\n🧠 Analyse du profil en cours...")
        interpretation = self.interpret_profile(all_reports)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON complet
        json_file = output_path / f'rapport_complet_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'images': all_reports,
                'interpretation_osint': interpretation
            }, f, ensure_ascii=False, indent=2)
        
        # Rapport texte avec interprétation
        txt_file = output_path / f'profil_osint_{timestamp}.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"🔍 PROFIL OSINT - {len(all_reports)} IMAGES ANALYSÉES\n")
            f.write(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write("="*60 + "\n\n")
            
            # INTERPRÉTATION
            f.write("🧠 INTERPRÉTATION DU PROFIL\n")
            f.write("="*60 + "\n\n")
            
            for category, insights in interpretation.items():
                category_names = {
                    'animaux': '🐾 ANIMAUX & COMPAGNIE',
                    'loisirs': '🎯 LOISIRS & HOBBIES',
                    'lifestyle': '🏠 STYLE DE VIE',
                    'mobilite': '🚗 MOBILITÉ',
                    'technologie': '💻 TECHNOLOGIE',
                    'localisation_probable': '📍 LOCALISATION',
                    'profil_socio': '👤 PROFIL SOCIAL'
                }
                f.write(f"{category_names.get(category, category.upper())}\n")
                f.write("-" * 60 + "\n")
                for insight in insights:
                    f.write(f"  • {insight}\n")
                f.write("\n")
            
            # Statistiques
            all_objects = []
            for r in all_reports:
                all_objects.extend(r['objets_detectes'])
            
            object_counts = Counter(all_objects)
            
            f.write("\n📊 STATISTIQUES DÉTECTÉES\n")
            f.write("="*60 + "\n")
            f.write(f"Images analysées: {len(all_reports)}\n")
            f.write(f"Objets uniques: {len(set(all_objects))}\n\n")
            
            f.write("Top 15 objets détectés:\n")
            for obj, count in object_counts.most_common(15):
                f.write(f"  • {obj}: {count}x\n")
        
        # Affichage console
        print("\n" + "="*60)
        print("🧠 PROFIL OSINT GÉNÉRÉ")
        print("="*60)
        
        for category, insights in interpretation.items():
            if insights:
                print(f"\n{category.upper()}:")
                for insight in insights:
                    print(f"  • {insight}")
        
        print("\n" + "="*60)
        print(f"✅ Analyse terminée - {len(all_reports)} images")
        print(f"💾 Rapports: {output_path}/")
        print(f"   - {json_file.name}")
        print(f"   - {txt_file.name}")
        
        return all_reports, interpretation


if __name__ == "__main__":
    osint = CloudPhotoOSINT()
    
    print("\n🗂️  ANALYSEUR OSINT AVEC INTERPRÉTATION")
    print("="*60)
    folder = input("📁 Chemin du dossier photos (ou 'skip'): ").strip()
    
    if folder.lower() != 'skip':
        osint.analyze_folder(folder)
    else:
        print("\n🌐 Mode test URL")
        result = osint.analyze("https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d")
        print(f"Objets: {result['objets_detectes']}")
        print(f"Description: {result['description_ia']}")