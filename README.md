# PureThermal 3 – FLIR Lepton Viewer

Cette application Python offre une interface graphique complète pour le module
thermique **PureThermal 3 – FLIR Lepton** sous Ubuntu 20. Elle permet de
visualiser le flux thermique, d’afficher une échelle de couleur avec les
températures correspondantes, de connaître la température minimale et maximale
du champ de vision, ainsi que la température sous le pointeur de la souris.
L’interface propose également des boutons pour enregistrer des photos et des
vidéos en choisissant la fréquence d’acquisition.

## Fonctionnalités

- Affichage temps réel du flux thermique avec palette de couleurs (Inferno).
- Échelle de couleur dynamique indiquant les températures min/max de la scène.
- Indication des températures minimale et maximale directement sur l’image.
- Pointeur contrôlé à la souris pour lire la température d’un pixel.
- Capture d’images fixes (PNG/JPG/BMP).
- Enregistrement vidéo avec choix de la fréquence (9 FPS ou 27 FPS par défaut).
- Enregistrement des vidéos au format AVI (codec XVID).

## Prérequis matériels et logiciels

- Carte PureThermal 3 avec module FLIR Lepton.
- Ubuntu 20 (ou distribution compatible).
- Python 3.8+.
- Accès aux périphériques USB (ajouter votre utilisateur au groupe `plugdev`
  si nécessaire : `sudo usermod -a -G plugdev $USER`).
- Bibliothèques système : `libusb-1.0-0`, `v4l-utils`, `ffmpeg`.

Installez d’abord les dépendances système :

```bash
sudo apt update
sudo apt install -y libusb-1.0-0-dev v4l-utils ffmpeg
```

Ensuite, créez un environnement virtuel Python et installez les dépendances
Python :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Remarque :** `pylibuvc` nécessite que `libusb` soit correctement installé.
> Si l’installation échoue, consultez la documentation du projet PureThermal
> pour compiler le module à partir des sources.

## Lancement de l’application

Activez l’environnement virtuel (si ce n’est pas déjà fait) puis lancez le
programme :

```bash
source .venv/bin/activate
python thermal_viewer.py
```

### Contrôles de l’interface

- **Fréquence vidéo :** sélectionnez la cadence d’acquisition avant de lancer un
  enregistrement vidéo.
- **Capture Photo :** enregistre une image colorisée dans le fichier de votre
  choix.
- **Démarrer l’enregistrement / Arrêter l’enregistrement :** crée une vidéo au format
  AVI en utilisant la fréquence sélectionnée.
- **Pointeur souris :** déplacez la souris au-dessus de l’image pour lire la
  température en bas à droite de l’interface.

Les vidéos et photos sont enregistrées dans le répertoire indiqué au moment de
la sauvegarde. Par défaut, l’application suggère un nom basé sur la date et
heure dans votre dossier personnel.

## Dépannage

- **Aucune caméra détectée :** vérifiez le câble USB, les permissions, et que
  la carte PureThermal 3 est alimentée.
- **Erreur `pylibuvc` introuvable :** assurez-vous d’avoir installé la
  dépendance (voir la section Prérequis). Si vous utilisez un environnement
  virtuel, activez-le avant le lancement.
- **Vidéo illisible :** installez `ffmpeg` ou un lecteur compatible AVI/XVID.

## Licence

Ce projet est fourni sans licence spécifique. Adaptez-le librement selon vos
besoins.
