# Projet de Prédiction de Flux de Véhicules Enrichi avec Données Météorologiques

## Description du Projet
Ce projet vise à prédire le flux de véhicules sur plusieurs points de collecte à l'aide de modèles de machine learning basés sur des données historiques de trafic et des données météorologiques. L'objectif principal est de développer, évaluer et optimiser des modèles prédictifs tels que LSTM, XGBoost et Random Forest pour fournir des prédictions précises à partir de caractéristiques prétraitées et enrichies. Le projet inclut des modèles préentraînés sur des périodes allant jusqu'à un mois et demi de données, permettant de capturer des tendances complexes et saisonnières.


## Installation et Configuration
Pour répliquer l'environnement sur Mac ou Unix, suivez les instructions ci-dessous :

1. Clonez ce dépôt.
2. Créez un environnement virtuel et activez-le :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt

## Utilisation
Pour exécuter les notebooks Jupyter, utilisez la commande suivante :
```bash
jupyter notebook
```
Pour lancer l'interface de prédiction, exécutez :
```bash
streamlit run ./interface.py
```

## Organisation et Explication des Fichiers et Dossiers

- **randomforest_1.5month_50**, **randomforest_1month_50**, **randomforest_3month_50** : Répertoires contenant les modèles Random Forest entraînés sur différentes périodes (1,5 mois, 1 mois et 3 mois).
- **xgboost_regressor_1month**, **xgboost_regressor_3month**, **xgboost_regressor_extended** : Répertoires contenant les modèles XGBoost entraînés sur différentes périodes (1 mois, 3 mois et une période étendue).
- **1month10.keras**, **1month5.keras**, **best_extended16.keras**, **best_extended19.keras** : Modèles LSTM sauvegardés avec les meilleures performances sur différentes périodes.
- **1month_meteo_data_en.csv**, **1month_meteo_data_so.csv**, **3month_meteo_data_en.csv**, **3month_meteo_data_so.csv**, **extended_meteo_data_en.csv**, **extended_meteo_data_so.csv** : Fichiers de données enrichis pour différents horizons de temps( extended signifie 1 mois et demi de donnée).
- **Camera_data.ipynb**, **radar.ipynb**, **tube.ipynb** : Notebooks pour l'analyse des données collectées respectivement par les caméras, les radars et les tubes (les fichiers traitent un peu près chaque fichier diffèremment).
- **traitement_data.ipynb** : Crée les ensembles de données finales avec l'intervalle temporel au choix.
- **Lstm.ipynb** : Notebook pour l'entraînement et l'évaluation du modèle LSTM.Ainsi, que l'algorithme de prédiction itérative et l'algorithme d'optimisation des hyperparamètres.
- **randomforest.ipynb**, **XGBoostt.ipynb** : Notebooks pour l'entraînement et l'évaluation des modèles Random Forest et XGBoost.
- **interface.py** : Interface Streamlit permet aux utilisateurs de charger des fichiers de modèles prédictifs, de scaler et des données pour générer des prédictions basées sur des modèles tels que LSTM, Random Forest et XGBoost, en utilisant des données météorologiques.
- **requirements.txt** : Liste des dépendances requises pour exécuter ce projet.
- **analyse_data.ipynb** : Contient quelques analyses effectués sur la data.
- **scaler_features_1month.pkl**, **scaler_features_3month.pkl**, **scaler_features_extended.pkl** : Scalers utilisés pour normaliser les caractéristiques des ensembles de données respectifs pour les modèles LSTM.
- **scaler_target_1month.pkl**, **scaler_target_3month.pkl**, **scaler_target_extended.pkl** : Scalers utilisés pour normaliser les cibles pour les modèles LSTM.
- **scaler_features_random_1month.pkl**, **scaler_features_random_3month.pkl**, **scaler_features_random_1.5month.pkl** : Scalers utilisés pour normaliser les caractéristiques des ensembles de données respectifs pour les modèles LSTM.


## Dépendances Principales
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `tensorflow`
- `meteostat`
- `matplotlib`
- `joblib`

## Note
Les ensembles de données prétraités incluent des colonnes telles que 2R, VL, PL et des caractéristiques météorologiques comme température, humidité et précipitations. Ces données enrichies permettent une meilleure précision des modèles prédictifs et facilitent leur adaptation aux tendances réelles et aux variations saisonnières.

Le code est souvent structuré sous forme de fonctions, permettant un contrôle facile des paramètres. Cela rend l'ensemble du projet réutilisable et modifiable pour d'autres applications et analyses.
