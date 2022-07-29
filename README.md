# Exemple d'interface pour l'analyse et la gestion des entrepôts de IDEA-groupe

## Exécuter le code
### Lancement de l'interface
L'interface repose sur un environnement python. Nous vous recommandons d'installer Anaconda. Depuis le terminal, vous devez effectuer les lignes de commandes suivantes : 
(En remplaçant ... par le chemin du répertoire)

On créer un environnement :

conda create -n idea-env python=3.8

conda activate idea-env

On installe les dépendances :

cd .../idea

pip install -r requirements.txt

Pour lancer l'interface, il suffira d’executer :

streamlit run interface.py

### Lancement des notebooks

Si ce n'est pas déjà fait, installer jupyter: 

conda install jupyter

Puis l’exécuter :

jupyter notebook

## Structure du projet 

- interface.py: le squelette de l'interface. Les fichiers suivants contiennent les sous-parties :
    -   Inventaire.py
    -   Description.py
    -   Productivité.py
    -   Rangement.py
    -   Recommandation.py
    Les classes implémentées dans ces fichiers héritent toutes de la classe présente dans affichage.py.

- Utilisation_simulateur.ipynb: un notebook permettant d'utiliser le simulateur de rangement présent    dans /Simulateur

- Simulateur/simulateur.py: Le simulateur de rangement d'entrepot, ainsi que différentes méthodes de rangement d'entrepot appelées 'agent'

- productivity_analysis/shop.py

- productivity_analysis/productivity.py

- Volumes_ref_magasins/Volume_cameleon.csv

/Data contient les données nécessaires

- /Data/Data_extraction_Reflex.ipynb sert à extraire les données pour utiliser le simulateur depuis le notebook. Ce notebook créé les fichiers suivants :
    stock_initial.csv
    portion_caisse.csv
    nb_par_caisse.csv
    analyse_Alexis.csv

- /Données_V2 contient les données originelles
- /Données_V2_propores contient les données au format requis par l'interface


