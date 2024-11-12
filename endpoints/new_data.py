from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import datetime
import os
import pandas as pd
from endpoints.auth import get_admin_user, User 

router = APIRouter()
class NewDataParams(BaseModel):
    base_url: str
    year: int

@router.get("/get_data")
async def get_last_year_data_url(params: NewDataParams, current_user: User = Depends(get_admin_user)):
    """Fonction pour extraire l'URL de téléchargement pour l'année donnée."""
    response = requests.get(base_url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)
    
    for link in links:
        if f"{year}" in link['href'] and link['href'].endswith('.csv'):
            return link['href']
    return None

@router.get("/download_data")
async def download_and_concatenate_data():
    # Détermine l'année précédente
    current_year = datetime.datetime.now().year
    last_year = current_year - 1

    # URL de la page des ressources
    base_url = "https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/#/resources"

    # Récupérer l'URL de téléchargement pour l'année précédente
    download_url = get_last_year_data_url(base_url, last_year)
    if not download_url:
        print(f"URL de téléchargement pour {last_year} introuvable.")
        return
    
    # Définir le dossier et le nom du fichier pour stocker les nouvelles données
    data_dir = "accidents_data"
    os.makedirs(data_dir, exist_ok=True)
    new_data_file = os.path.join(data_dir, f"accidents_{last_year}.csv")
    
    # Télécharger les données de l'année passée si elles ne sont pas déjà présentes
    if not os.path.exists(new_data_file):
        print(f"Téléchargement des données de {last_year} depuis {download_url}...")
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            with open(new_data_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Données de {last_year} téléchargées et enregistrées sous {new_data_file}")
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors du téléchargement des données : {e}")
            return
    else:
        print(f"Les données de {last_year} existent déjà.")

    # Charger les nouvelles données
    new_data = pd.read_csv(new_data_file)
    
    # Charger les données existantes ou créer un nouveau fichier principal s'il n'existe pas
    main_data_file = os.path.join(data_dir, "accidents_historique.csv")
    if os.path.exists(main_data_file):
        print("Chargement des données existantes...")
        main_data = pd.read_csv(main_data_file)
        # Concaténer les nouvelles données aux données existantes
        print("Concaténation des nouvelles données aux données existantes...")
        main_data = pd.concat([main_data, new_data], ignore_index=True)
    else:
        print("Aucun fichier de données existant. Création d'un nouveau fichier principal...")
        main_data = new_data

    # Enregistrer les données mises à jour dans le fichier principal
    main_data.to_csv(main_data_file, index=False)
    print(f"Données mises à jour enregistrées sous {main_data_file}")