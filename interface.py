import os
import tempfile
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from meteostat import Point, Hourly
from datetime import datetime, timedelta
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore

def get_default_start_date(df):
    return df['Datetime'].max() - pd.Timedelta(days=7)

# Fonction pour générer les caractéristiques nécessaires pour les prédictions
def generate_features(start_date, end_date, location):
    # Convertir les dates en objets datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Récupérer les données météorologiques depuis Meteostat
    meteo_data = Hourly(location, start, end).fetch()

    # Prétraitement des données météo
    meteo_data['DayOfWeek'] = meteo_data.index.dayofweek
    meteo_data['Hour'] = meteo_data.index.hour
    meteo_data['DayOfMonth'] = meteo_data.index.day
    meteo_data['Month'] = meteo_data.index.month

    # Renommer les colonnes pour correspondre aux noms utilisés dans le modèle
    meteo_data.rename(columns={
        'temp': 'temperature(degC)',
        'dwpt': 'point_de_rosee(degC)',
        'rhum': 'humidite(%)',
        'prcp': 'precipitations(mm)',
        'snow': 'neige(mm)',
        'wdir': 'vent_direction(deg)',
        'wspd': 'vent_moyen(km/h)',
        'wpgt': 'rafale_vent_max(km/h)',
        'pres': 'pression(hPa)',
        'tsun': 'ensoleillement(H)'
    }, inplace=True)

    required_columns = ['Datetime', 'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'temperature(degC)',
                        'point_de_rosee(degC)', 'humidite(%)', 'precipitations(mm)',
                        'neige(mm)', 'vent_direction(deg)', 'vent_moyen(km/h)',
                        'rafale_vent_max(km/h)', 'pression(hPa)']

    # Assurer que 'Datetime' est une colonne
    meteo_data['Datetime'] = meteo_data.index

    # Garder uniquement les colonnes nécessaires
    meteo_data = meteo_data[required_columns]

    # Remplir les valeurs manquantes
    meteo_data.fillna(method='ffill', inplace=True)

    return meteo_data.reset_index(drop=True)

def predict_random_forest_classifier(start_date, end_date, models_folder, scaler, location):
    features = ['Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'temperature(degC)', 'point_de_rosee(degC)',
                'humidite(%)', 'precipitations(mm)', 'neige(mm)', 'vent_direction(deg)',
                'vent_moyen(km/h)', 'rafale_vent_max(km/h)', 'pression(hPa)']

    # Génération des caractéristiques pour les dates de prédiction
    features_df = generate_features(start_date, end_date, location)
    
    # Normalisation des données d'entrée avec le scaler chargé
    X_scaled = pd.DataFrame(scaler.transform(features_df[features]), columns=features)
    X_scaled['Datetime'] = features_df['Datetime']

    # Initialisation du DataFrame pour stocker les prédictions
    predictions = pd.DataFrame()
    predictions['Datetime'] = features_df['Datetime']

    # Liste des fichiers de modèles dans le dossier spécifié
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.joblib')]

    for model_file in model_files:
        model_path = os.path.join(models_folder, model_file)
        model = joblib.load(model_path)
        
        # Extraction du nom de la colonne cible à partir du nom du fichier
        target_col = model_file.replace('random_forest_', '').replace('.joblib', '')

        # Prédiction des classes
        y_pred = model.predict(X_scaled[features])

        # Ajout des prédictions au DataFrame
        predictions[f'random_forest_{target_col}'] = y_pred

    return predictions
# # Fonction pour prédire avec Random Forest Classifier
# def predict_random_forest_classifier(start_date, end_date, models_folder, scalers_folder, location):
#     features = ['Hour','DayOfWeek', 'DayOfMonth','Month', 'temperature(degC)', 'point_de_rosee(degC)',
#                 'humidite(%)', 'precipitations(mm)', 'neige(mm)', 'vent_direction(deg)',
#                 'vent_moyen(km/h)', 'rafale_vent_max(km/h)', 'pression(hPa)']

#     # Génération des caractéristiques pour les dates de prédiction
#     features_df = generate_features(start_date, end_date, location)

#     # Liste des fichiers de modèles dans le dossier spécifié
#     model_files = [f for f in os.listdir(models_folder) if f.endswith('.joblib')]

#     predictions = pd.DataFrame()
#     predictions['Datetime'] = features_df['Datetime']

#     for model_file in model_files:
#         model_path = os.path.join(models_folder, model_file)
#         model = joblib.load(model_path)
        
#         # Extraction du nom de la colonne cible à partir du nom du fichier
#         target_col = model_file.replace('random_forest_', '').replace('.joblib', '')

#         # Chargement du scaler correspondant
#         scaler_filename = os.path.join(scalers_folder, f'scaler_{target_col}.pkl')
#         scaler = joblib.load(scaler_filename)

#         # Préparation des caractéristiques
#         X = features_df[features]
#         X_scaled = scaler.transform(X)

#         # Prédiction des classes
#         y_pred = model.predict(X_scaled)

#         # Ajout des prédictions au DataFrame
#         predictions[f'random_forest_{target_col}'] = y_pred

#     return predictions

# Fonction pour prédire à partir du modèle XGBoost
def predict_xgboost_regressor(start_date, end_date, models_folder, location):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Récupérer les données météorologiques
    meteo_data = Hourly(location, start, end).fetch()

    # Prétraitement des données météo
    meteo_data['DayOfWeek'] = meteo_data.index.dayofweek
    meteo_data['Hour'] = meteo_data.index.hour
    meteo_data['DayOfMonth'] = meteo_data.index.day
    meteo_data['Month'] = meteo_data.index.month

    # Renommer les colonnes pour correspondre aux noms utilisés dans le modèle
    meteo_data.rename(columns={
        'temp': 'temperature(degC)',
        'dwpt': 'point_de_rosee(degC)',
        'rhum': 'humidite(%)',
        'prcp': 'precipitations(mm)',
        'snow': 'neige(mm)',
        'wdir': 'vent_direction(deg)',
        'wspd': 'vent_moyen(km/h)',
        'wpgt': 'rafale_vent_max(km/h)',
        'pres': 'pression(hPa)',
        'tsun': 'ensoleillement(H)'
    }, inplace=True)

    required_columns = ['Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'temperature(degC)', 
                        'point_de_rosee(degC)', 'humidite(%)', 'precipitations(mm)', 
                        'neige(mm)', 'vent_direction(deg)', 'vent_moyen(km/h)', 
                        'rafale_vent_max(km/h)', 'pression(hPa)']
    
    meteo_data = meteo_data[required_columns]

    # Création du DataFrame pour les prédictions
    predictions_df = pd.DataFrame(index=meteo_data.index)
    predictions_df['Datetime'] = meteo_data.index

    # Charger et appliquer chaque modèle XGBoost sauvegardé
    for model_filename in os.listdir(models_folder):
        if model_filename.endswith('.joblib'):
            column_name = model_filename.replace('.joblib', '')
            model = joblib.load(os.path.join(models_folder, model_filename))
            predictions = model.predict(meteo_data)
            predictions_df[column_name] = predictions
            non_datetime_columns = predictions_df.select_dtypes(exclude=['datetime']).columns
            predictions_df[non_datetime_columns] = predictions_df[non_datetime_columns].round().astype('Int64')

    return predictions_df

# Fonction pour initialiser la date de début à 7 jours avant la dernière date du fichier
def get_default_start_date(df, default_days=7):
    if 'Datetime' in df.columns:
        last_date = df['Datetime'].max()
        return last_date - timedelta(days=default_days)
    else:
        return datetime.now() - timedelta(days=default_days)


def prepare_initial_data(df, required_columns, history_size):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes suivantes sont manquantes dans les données d'entrée: {missing_cols}")
    
    df = df[required_columns]
    initial_data = df.tail(history_size)
    return initial_data

def iterative_prediction_with_meteo(model, initial_data, num_days_to_predict, history_size, target_size, scaler_features, scaler_targets, location):
    
    num_predictions = num_days_to_predict * 24 // target_size
    
    all_predictions = []

    for i in range(num_predictions):
        last_week_data = initial_data[-history_size:]  # Prendre la dernière semaine de données
        if 'Datetime' in last_week_data.columns:
            last_week_data_without_datetime = last_week_data.drop(columns=['Datetime'])
        last_datetime = pd.to_datetime(last_week_data['Datetime'].iloc[-1])

        if pd.isna(last_datetime):
            raise ValueError("La colonne 'Datetime' contient une valeur NaT (Not a Timestamp) ou est manquante.")
        #print("Last week data:")
        #print(last_week_data[['Datetime']].tail(10))  # Affichez les 10 dernières valeurs de 'Datetime' pour vérifier les données

        prediction_dates = pd.date_range(start=last_datetime + timedelta(hours=1), periods=target_size, freq='h')

        features_seq = np.array(last_week_data_without_datetime).reshape((1, history_size, last_week_data_without_datetime.shape[1]))
        
        predicted_targets_scaled = model.predict(features_seq)
        # predicted_targets_scaled = predicted_targets_scaled.reshape(target_size, -1)
        #print(f'Forme initiale des prédictions: {predicted_targets_scaled.shape}')
        predicted_targets_scaled = predicted_targets_scaled.reshape(target_size, -1)
        #print(f'Forme après reshape: {predicted_targets_scaled.shape}')
        predicted_targets = scaler_targets.inverse_transform(predicted_targets_scaled)
        predicted_columns = scaler_targets.feature_names_in_  # Assurez-vous que cela contient les noms de colonnes corrects

        
        predicted_targets_df = pd.DataFrame(predicted_targets, columns=predicted_columns)#initial_data.columns[-predicted_targets.shape[1]:])  
    
        predicted_targets_df['Datetime'] = prediction_dates

        predicted_targets_scaled_df = pd.DataFrame(predicted_targets_scaled, index=prediction_dates, columns=scaler_targets.feature_names_in_)        
        predicted_targets_scaled_df['Datetime'] = prediction_dates

        start = prediction_dates[0]
        end = prediction_dates[-1]
        meteo_data = Hourly(location, start, end).fetch()
        
        meteo_data['DayOfWeek'] = meteo_data.index.dayofweek
        meteo_data['Hour'] = meteo_data.index.hour
        meteo_data['DayOfMonth'] = meteo_data.index.day
        meteo_data['Month'] = meteo_data.index.month
        

        meteo_data.rename(columns={
            'temp': 'temperature(degC)',
            'dwpt': 'point_de_rosee(degC)',
            'rhum': 'humidite(%)',
            'prcp': 'precipitations(mm)',
            'snow': 'neige(mm)',
            'wdir': 'vent_direction(deg)',
            'wspd': 'vent_moyen(km/h)',
            'wpgt': 'rafale_vent_max(km/h)',
            'pres': 'pression(hPa)',
            'tsun': 'ensoleillement(H)'
        }, inplace=True)

        meteo_data['Datetime'] = meteo_data.index
        meteo_data.reset_index(drop=True, inplace=True)
        meteo_data.fillna(0, inplace=True)
    
        colonnes_souhaitees = list(scaler_features.feature_names_in_)


        colonnes_presentes = [col for col in colonnes_souhaitees if col in meteo_data.columns]
        meteo_data = meteo_data[colonnes_presentes]

        colonnes_indesirables = ['coco']  # Ajouter d'autres colonnes si nécessaire
        meteo_data = meteo_data.drop(columns=[col for col in colonnes_indesirables if col in initial_data.columns])
        #print(meteo_data)
        meteo_data_scaled = scaler_features.transform(meteo_data)

        meteo_df = pd.DataFrame(meteo_data_scaled, columns=meteo_data.columns)
        meteo_df['Datetime'] = prediction_dates

        next_input_df = pd.merge(meteo_df, predicted_targets_scaled_df, left_on='Datetime', right_index=True,how = 'left')
        

        expected_columns = list(scaler_features.feature_names_in_) + list(scaler_targets.feature_names_in_)
        #next_input_df = next_input_df[expected_columns]
        next_input_df = next_input_df[expected_columns]
        next_input_df.insert(0, 'Datetime', prediction_dates)

        initial_data = pd.concat([initial_data, next_input_df], ignore_index=True)
        all_predictions.append(pd.concat([meteo_df[['Datetime']], predicted_targets_df], axis=1))
    
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    target_columns = list(scaler_targets.feature_names_in_)
    final_predictions[target_columns] = final_predictions[target_columns].round().abs().astype(int)
    final_predictions = final_predictions.iloc[:, :-1]
    return final_predictions


def predict_with_history_and_offset(df, model, scaler_features, scaler_targets, offset_days, num_days_to_predict):
    history_size = 168  # 7 jours d'historique en heures (modifiable)
    offset_size = offset_days * 24  # Calculer l'offset en heures

    # Sélection des données basées sur l'offset et l'historique
    relevant_data = df.tail(history_size + offset_size)
    historical_data = relevant_data.iloc[:history_size]

    # Préparer les features et targets
    features = historical_data[scaler_features.feature_names_in_]
    targets = historical_data[scaler_targets.feature_names_in_]
    
    # Transformer les données avec les scalers
    features_scaled = scaler_features.transform(features)
    targets_scaled = scaler_targets.transform(targets)
    
    # Créer un DataFrame pour les données transformées
    historical_data_scaled = pd.DataFrame(np.hstack((features_scaled, targets_scaled)), 
                                          columns=list(scaler_features.feature_names_in_) + list(scaler_targets.feature_names_in_))
    historical_data_scaled['Datetime'] = historical_data['Datetime'].values
    
    # Effectuer la prédiction
    predictions = iterative_prediction_with_meteo(
        model=model, 
        initial_data=historical_data_scaled,  # Semaine d'historique sans 'Datetime'
        num_days_to_predict=num_days_to_predict,  # Nombre de jours à prédire
        history_size=history_size,  # 7 jours d'historique (168 heures)
        target_size=1,  # Taille de prédiction : 1 heure à la fois
        scaler_features=scaler_features, 
        scaler_targets=scaler_targets, 
        location=Point(44.8069, -0.6133, 20)  # Talence, près de Bordeaux
    )
    
    return predictions

# Streamlit UI
st.title("Prédictions des Modèles de Trafic")

# Sélection du type de modèle
model_type = st.selectbox("Choisir le type de modèle", ["LSTM", "Random Forest", "XGBoost"])

if model_type == "XGBoost":
    models_folder = st.text_input("Chemin du dossier des modèles XGBoost", value="./xgboost_regressor_1month")

    # Téléchargement du fichier CSV pour les données historiques
    data_file = st.file_uploader("Choisir un fichier de données CSV (optionnel)", type="csv")

    # Si un fichier CSV est fourni, initialiser la date de début 7 jours avant la dernière date du fichier
    if data_file:
        df = pd.read_csv(data_file, parse_dates=['Datetime'])
        default_start_date = get_default_start_date(df)
        st.write("Aperçu des données :")
        st.write(df.head())
    else:
        default_start_date = datetime.now() - timedelta(days=7)

    start_date = st.date_input("Date de début", value=default_start_date)
    end_date = st.date_input("Date de fin", value=start_date + timedelta(days=15))

    location = Point(44.8069, -0.6133, 20)

    if "previous_end_date" not in st.session_state or st.session_state.previous_end_date != end_date:
        st.session_state.predictions = predict_xgboost_regressor(start_date=start_date, end_date=end_date, models_folder=models_folder, location=location)
        st.session_state.previous_end_date = end_date

    predictions = st.session_state.predictions
    st.write("Prédictions :")
    st.write(predictions)

    # Affichage de la prédiction selon le motif avant le tableau
    st.subheader("Filtrer les prédictions par motif")
    filtered_columns = []
    
    motif = st.text_input("Saisissez le motif pour filtrer les colonnes de prédictions (exemple: 2R-P01 ou 2R) :")
    if motif:
        filtered_columns = [col for col in predictions.columns if col != 'Datetime' and motif.lower() in col.lower()]

    if filtered_columns:
        st.write(f"Affichage des colonnes de prédictions basées sur le motif : {motif}")

        for col in filtered_columns:
            st.write(f"Graphique pour la colonne : {col}")
            st.line_chart(predictions.set_index('Datetime')[[col]])  # Afficher chaque colonne avec traits (sans points)
    else:
        st.write("Aucune colonne ne correspond au motif.")

    # Affichage du tableau comparatif après le motif
    if data_file:
        st.subheader("Comparaison des prédictions avec les valeurs historiques (tableaux)")

        # Sélectionner les colonnes historiques (réelles) et leurs prédictions
        historical_data = df[['Datetime'] + [col for col in df.columns if col.startswith('VL') or col.startswith('PL') or col.startswith('2R')]]
        merged_data = pd.merge(historical_data, predictions, on='Datetime', how='inner')

        # Réorganisation des colonnes réelles suivies des prédictions correspondantes
        columns_order = ['Datetime']  # Commence par 'Datetime'
        for col in historical_data.columns:
            if col != 'Datetime':
                columns_order.append(col)  # Ajout de la colonne réelle
                pred_col = f"xgboost_{col}"  # Colonne de prédiction correspondante
                if pred_col in merged_data.columns:
                    columns_order.append(pred_col)  # Ajout de la colonne de prédiction

        # Réorganiser les colonnes dans l'ordre
        merged_data = merged_data[columns_order]

        # Afficher le tableau comparatif
        st.write("Tableau comparatif : valeurs historiques et prédictions")
        st.write(merged_data)

    # Comparaison des colonnes réelles et prédictions par motif (graphes)
    if data_file:
        st.subheader("Comparer les prédictions et les valeurs historiques (graphes)")

        with st.form(key="comparison_form"):
            comparison_motif = st.text_input("Saisissez le motif pour comparer les colonnes (exemple: 2R-P01 ou 2R) :")
            compare_button = st.form_submit_button(label="Afficher les graphes de comparaison")

        if compare_button and comparison_motif:
            comparison_columns = []
            for col in historical_data.columns:
                if col.startswith(comparison_motif) and f"xgboost_{col}" in merged_data.columns:
                    comparison_columns.append(col)  # Valeurs historiques
                    comparison_columns.append(f"xgboost_{col}")  # Prédictions

            if comparison_columns:
                for i in range(0, len(comparison_columns), 2):
                    actual_col = comparison_columns[i]
                    predicted_col = comparison_columns[i+1]
                    st.write(f"Comparaison entre {actual_col} (valeur réelle) et {predicted_col} (prédiction)")

                    # Créer un graphique avec Matplotlib, style simple
                    fig, ax = plt.subplots(figsize=(10, 5))

                    ax.plot(merged_data['Datetime'], merged_data[actual_col], label=actual_col, color='blue')
                    ax.plot(merged_data['Datetime'], merged_data[predicted_col], label=predicted_col, color='lightblue')

                    # Ajouter les labels des axes
                    ax.set_xlabel("Date", fontsize=12)
                    ax.set_ylabel("Flux des véhicules", fontsize=12)
                    ax.set_title(f"Comparaison des flux : {actual_col} vs {predicted_col}", fontsize=14)

                    # Ajouter une légende
                    ax.legend()

                    # Rotation des dates pour qu'elles soient plus lisibles
                    plt.xticks(rotation=45)

                    # Simplifier le style, enlever les bordures autour du graphe
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(True)

                    ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
                    ax.xaxis.grid(False)

                    # Afficher le graphique dans Streamlit
                    st.pyplot(fig)
            else:
                st.write("Aucune colonne ne correspond au motif pour la comparaison.")

# if model_type == "XGBoost":
#     models_folder = st.text_input("Chemin du dossier des modèles XGBoost", value="./xgboost_regressor_1month")

#     # Téléchargement du fichier CSV pour les données historiques
#     data_file = st.file_uploader("Choisir un fichier de données CSV (optionnel)", type="csv")

#     # Si un fichier CSV est fourni, initialiser la date de début 7 jours avant la dernière date du fichier
#     if data_file:
#         df = pd.read_csv(data_file, parse_dates=['Datetime'])
#         default_start_date = get_default_start_date(df)
#         st.write("Aperçu des données :")
#         st.write(df.head())
#     else:
#         default_start_date = datetime.now() - timedelta(days=7)

#     start_date = st.date_input("Date de début", value=default_start_date)
#     end_date = st.date_input("Date de fin", value=start_date + timedelta(days=15))

#     location = Point(44.8069, -0.6133, 20)

#     if "previous_end_date" not in st.session_state or st.session_state.previous_end_date != end_date:
#         st.session_state.predictions = predict_xgboost_regressor(start_date=start_date, end_date=end_date, models_folder=models_folder, location=location)
#         st.session_state.previous_end_date = end_date

#     predictions = st.session_state.predictions
#     st.write("Prédictions :")
#     st.write(predictions)

#     if data_file:
#         st.subheader("Comparaison des prédictions avec les valeurs historiques (tableaux)")

#         # Sélectionner les colonnes historiques (réelles) et leurs prédictions
#         historical_data = df[['Datetime'] + [col for col in df.columns if col.startswith('VL') or col.startswith('PL') or col.startswith('2R')]]
#         merged_data = pd.merge(historical_data, predictions, on='Datetime', how='inner')

#         # Réorganisation des colonnes réelles suivies des prédictions correspondantes
#         columns_order = ['Datetime']  # Commence par 'Datetime'
#         for col in historical_data.columns:
#             if col != 'Datetime':
#                 columns_order.append(col)  # Ajout de la colonne réelle
#                 pred_col = f"xgboost_{col}"  # Colonne de prédiction correspondante
#                 if pred_col in merged_data.columns:
#                     columns_order.append(pred_col)  # Ajout de la colonne de prédiction

#         # Réorganiser les colonnes dans l'ordre
#         merged_data = merged_data[columns_order]

#         # Afficher le tableau comparatif
#         st.write("Tableau comparatif : valeurs historiques et prédictions")
#         st.write(merged_data)

#     # Partie graphique (garde la partie de recherche par motif pour les graphes)
#     st.subheader("Filtrer les prédictions par motif")
#     filtered_columns = []
    
#     motif = st.text_input("Saisissez le motif pour filtrer les colonnes de prédictions (exemple: 2R-P01 ou 2R) :")
#     if motif:
#         filtered_columns = [col for col in predictions.columns if col != 'Datetime' and motif.lower() in col.lower()]

#     if filtered_columns:
#         st.write(f"Affichage des colonnes de prédictions basées sur le motif : {motif}")

#         for col in filtered_columns:
#             st.write(f"Graphique pour la colonne : {col}")
#             st.line_chart(predictions.set_index('Datetime')[[col]])  # Afficher chaque colonne avec traits (sans points)
#     else:
#         st.write("Aucune colonne ne correspond au motif.")

#     # Comparaison des colonnes réelles et prédictions par motif
#     if data_file:
#         st.subheader("Comparer les prédictions et les valeurs historiques (graphes)")

#         with st.form(key="comparison_form"):
#             comparison_motif = st.text_input("Saisissez le motif pour comparer les colonnes (exemple: 2R-P01 ou 2R) :")
#             compare_button = st.form_submit_button(label="Afficher les graphes de comparaison")

#         if compare_button and comparison_motif:
#             comparison_columns = []
#             for col in historical_data.columns:
#                 if col.startswith(comparison_motif) and f"xgboost_{col}" in merged_data.columns:
#                     comparison_columns.append(col)  # Valeurs historiques
#                     comparison_columns.append(f"xgboost_{col}")  # Prédictions

#             if comparison_columns:
#                 for i in range(0, len(comparison_columns), 2):
#                     actual_col = comparison_columns[i]
#                     predicted_col = comparison_columns[i+1]
#                     st.write(f"Comparaison entre {actual_col} (valeur réelle) et {predicted_col} (prédiction)")
#                     st.line_chart(merged_data.set_index('Datetime')[[actual_col, predicted_col]])
#             else:
#                 st.write("Aucune colonne ne correspond au motif pour la comparaison.")

elif model_type == "LSTM":
    # Créer un formulaire pour gérer les entrées utilisateur
    with st.form(key="lstm_form"):
        model_file = st.file_uploader("Choisir le fichier du modèle LSTM", type="keras")
        scaler_features_file = st.file_uploader("Choisir le scaler des features", type="pkl")
        scaler_targets_file = st.file_uploader("Choisir le scaler des targets", type="pkl")


        num_days_to_predict = st.number_input("Nombre de jours à prédire", min_value=1, value=7)

        data_file = st.file_uploader("Choisir un fichier de données CSV", type="csv")

        predict_button = st.form_submit_button(label="Lancer la prédiction")

    if predict_button and data_file and model_file and scaler_features_file and scaler_targets_file:
        df = pd.read_csv(data_file, parse_dates=['Datetime'])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model_file:
            temp_model_file.write(model_file.getbuffer())
            temp_model_path = temp_model_file.name

        model = load_model(temp_model_path)
        scaler_features = joblib.load(scaler_features_file)
        scaler_targets = joblib.load(scaler_targets_file)
        location = Point(44.8069, -0.6133, 20)  # Talence, près de Bordeaux

        st.session_state['model'] = model
        st.session_state['scaler_features'] = scaler_features
        st.session_state['scaler_targets'] = scaler_targets

        required_columns = ['Datetime'] + list(scaler_features.feature_names_in_) + list(scaler_targets.feature_names_in_)

        initial_data = prepare_initial_data(df, required_columns, history_size=168)

        features = initial_data[scaler_features.feature_names_in_]
        features = scaler_features.transform(features)
        targets = initial_data[scaler_targets.feature_names_in_]
        targets = scaler_targets.transform(targets)

        initial_data = pd.DataFrame(np.hstack((features, targets)), columns=list(scaler_features.feature_names_in_) + list(scaler_targets.feature_names_in_))
        initial_data['Datetime'] = df['Datetime'].tail(168).reset_index(drop=True)

        initial_data = initial_data[['Datetime'] + list(scaler_features.feature_names_in_) + list(scaler_targets.feature_names_in_)]

        predictions = iterative_prediction_with_meteo(
            model=model, 
            initial_data=initial_data, 
            num_days_to_predict=num_days_to_predict, 
            history_size=168, 
            target_size=1, 
            scaler_features=scaler_features, 
            scaler_targets=scaler_targets, 
            location=location
        )
        print("Prédictions avant stockage:", predictions)
        st.session_state.predictions = predictions
        print("Prédictions dans session_state:", st.session_state.predictions)


    # Afficher les prédictions depuis session_state si elles existent
    if "predictions" in st.session_state:
        predictions = st.session_state.predictions
        st.write("Prédictions :")
        st.write(predictions)

        # Saisir un motif pour filtrer les colonnes des prédictions
        motif = st.text_input("Saisissez le motif pour filtrer les colonnes de prédictions (exemple: 2R-P01 ou 2R) :")

        if motif:
            filtered_columns = [col for col in predictions.columns if col != 'Datetime' and col.startswith(motif)]
            if filtered_columns:
                # Afficher les graphes pour les colonnes correspondantes au motif
                for col in filtered_columns:
                    st.write(f"Graphique pour la colonne : {col}")
                    st.line_chart(predictions.set_index('Datetime')[[col]])
            else:
                st.write("Aucune colonne ne correspond au motif.")
    

    st.subheader("Options supplémentaires")
    # Offset et jours supplémentaires
    offset_days = st.number_input("Nombre de jours à décaler (offset)", min_value=0, max_value=10, value=0, key="offset_days")
    num_days_to_predict = st.number_input("Nombre de jours à prédire", min_value=1, max_value=30, value=7, key="num_days_to_predict")

    # Télécharger un fichier CSV de données réelles pour la comparaison
    real_data_file = st.file_uploader("Choisir un fichier de données réelles CSV pour comparaison", type="csv")

    if real_data_file and "model" in st.session_state and "scaler_features" in st.session_state and "scaler_targets" in st.session_state:
        model = st.session_state['model']
        scaler_features = st.session_state['scaler_features']
        scaler_targets = st.session_state['scaler_targets']
        real_data = pd.read_csv(real_data_file, parse_dates=['Datetime'])

        # Ne recalculer que si l'offset ou la durée de prédiction changent
        if "previous_num_days_to_predict" not in st.session_state or st.session_state.previous_num_days_to_predict != num_days_to_predict or st.session_state.previous_offset_days != offset_days:
            st.session_state.new_predictions = predict_with_history_and_offset(real_data, model, scaler_features, scaler_targets, offset_days, num_days_to_predict)
            st.session_state.previous_num_days_to_predict = num_days_to_predict
            st.session_state.previous_offset_days = offset_days

        # Affichage des résultats (pas de recalcul lors de la saisie du motif)
        if "new_predictions" in st.session_state:
            new_predictions = st.session_state.new_predictions
            
            st.subheader("Comparaison des prédictions avec les données réelles")
            # Effectuer une fusion 'left' pour conserver toutes les prédictions et ajouter les valeurs réelles correspondantes si elles existent
            merged_data = pd.merge(new_predictions, real_data, on='Datetime', how='left', suffixes=('_pred', '_real'))

            # Renommer les colonnes pour enlever les suffixes et organiser les colonnes côte à côte
            renamed_columns = {}
            for col in merged_data.columns:
                if '_real' in col:
                    renamed_columns[col] = col.replace('_real', '')  # Supprimer le suffixe '_real'
                elif '_pred' in col:
                    renamed_columns[col] = col.replace('_pred', ' (prédiction)')  # Supprimer le suffixe '_pred' et ajouter "(prédiction)"
            merged_data.rename(columns=renamed_columns, inplace=True)

            # Réorganiser les colonnes pour avoir les paires réel-prédiction côte à côte
            columns_order = ['Datetime']  # Commencer avec 'Datetime'
            for col in merged_data.columns:
                if col != 'Datetime' and ' (prédiction)' not in col:
                    columns_order.append(col)  # Ajouter la colonne réelle
                    pred_col = col + ' (prédiction)'
                    if pred_col in merged_data.columns:
                        columns_order.append(pred_col)  # Ajouter la prédiction correspondante

            merged_data = merged_data[columns_order]  # Réorganiser dans l'ordre
            st.write("Tableau comparatif : valeurs réelles et prédictions")
            st.write(merged_data)

        # Afficher toute la prédiction et les valeurs historiques sur la même période
        comparison_motif = st.text_input("Saisissez le motif pour comparer les colonnes des prédictions et des valeurs réelles (exemple: 2R-P01 ou 2R) :")

        if comparison_motif:
            # Vérification pour trouver les colonnes réelles et de prédictions correspondantes
            filtered_columns_real = [col for col in merged_data.columns if comparison_motif.lower() in col.lower() and ' (prédiction)' not in col and col != 'Datetime']
            filtered_columns_pred = [col for col in merged_data.columns if comparison_motif.lower() in col.lower() and ' (prédiction)' in col and col != 'Datetime']

            # Vérifier que les colonnes filtrées existent avant d'essayer de les afficher
            if filtered_columns_real and filtered_columns_pred:
                for real_col, pred_col in zip(filtered_columns_real, filtered_columns_pred):
                    st.write(f"Comparaison entre {real_col} (réel) et {pred_col} (prédiction)")

                    # Préparer le DataFrame pour le graphique
                    comparison_df = merged_data[['Datetime', real_col, pred_col]].set_index('Datetime')

                    # Créer un graphique avec Matplotlib
                    fig, ax = plt.subplots(figsize=(10, 5))

                    ax.plot(comparison_df.index, comparison_df[real_col], label=real_col, color='blue')
                    ax.plot(comparison_df.index, comparison_df[pred_col], label=pred_col, color='lightblue')

                    # Ajouter les labels des axes
                    ax.set_xlabel("Date", fontsize=12)
                    ax.set_ylabel("Flux des véhicules", fontsize=12)
                    ax.set_title(f"Comparaison des flux : {real_col} vs {pred_col}", fontsize=14)

                    # Ajouter une légende
                    ax.legend()

                    # Rotation des dates pour qu'elles soient plus lisibles
                    plt.xticks(rotation=45)

                    # Simplifier le style, enlever les bordures inutiles
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(True)

                    ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
                    ax.xaxis.grid(False)

                    # Afficher le graphique dans Streamlit
                    st.pyplot(fig)
            else:
                st.warning("Aucune colonne ne correspond au motif. Veuillez vérifier que les colonnes existent.")







if model_type == "Random Forest":
    models_folder = st.text_input("Chemin du dossier des modèles Random Forest", value="./randomforest_1month_50")

    # Ajout du fichier pour le scaler (drag and drop)
    scaler_file = st.file_uploader("Charger le fichier du scaler", type="pkl")

    # Téléchargement du fichier CSV pour les données historiques
    data_file = st.file_uploader("Choisir un fichier de données CSV (optionnel)", type="csv")

    # Si un fichier CSV est fourni, initialiser la date de début 7 jours avant la dernière date du fichier
    if data_file:
        df = pd.read_csv(data_file, parse_dates=['Datetime'])
        default_start_date = get_default_start_date(df)
        st.write("Aperçu des données :")
        st.write(df.head())
    else:
        default_start_date = datetime.now() - timedelta(days=7)

    if "previous_models_folder" not in st.session_state:
        st.session_state.previous_models_folder = models_folder


    start_date = st.date_input("Date de début", value=default_start_date)
    end_date = st.date_input("Date de fin", value=start_date + timedelta(days=15))

    location = Point(44.8069, -0.6133, 20)

    # Si le scaler est téléchargé
    if scaler_file:
        scaler = joblib.load(scaler_file)

        if st.session_state.previous_models_folder != models_folder:
            st.session_state.predictions = predict_random_forest_classifier(
                start_date=start_date,
                end_date=end_date,
                models_folder=models_folder,
                scaler=scaler,
                location=location
            )
            st.session_state.previous_models_folder = models_folder

        if "previous_end_date" not in st.session_state or st.session_state.previous_end_date != end_date or st.session_state.previous_models_folder != models_folder:
            st.session_state.predictions = predict_random_forest_classifier(
                start_date=start_date,
                end_date=end_date,
                models_folder=models_folder,
                scaler=scaler,
                location=location
            )
            st.session_state.previous_end_date = end_date
            st.session_state.previous_models_folder = models_folder

        predictions = st.session_state.predictions
        st.write("Prédictions :")
        st.write(predictions)

        st.subheader("Filtrer les prédictions par motif")
        filtered_columns = []
        # Entrer le motif pour filtrer les colonnes à afficher
        motif = st.text_input("Saisissez le motif pour filtrer les colonnes de prédictions (exemple: 2R-P01 ou 2R) :")
        if motif:
            filtered_columns = [col for col in predictions.columns if col != 'Datetime' and motif.lower() in col.lower()]

        if filtered_columns:
            st.write(f"Affichage des colonnes de prédictions basées sur le motif : {motif}")

            for col in filtered_columns:
                st.write(f"Graphique pour la colonne : {col}")
                st.line_chart(predictions.set_index('Datetime')[[col]])  # Afficher chaque colonne avec traits (sans points)
        else:
            st.write("Aucune colonne ne correspond au motif.")

        if data_file:
            st.subheader("Comparaison des prédictions avec les valeurs historiques (tableaux)")

            # Sélection des colonnes cibles
            target_columns = [col for col in df.columns if col.startswith('VL') or col.startswith('PL') or col.startswith('2R')]
            historical_data = df[['Datetime'] + target_columns]

            # Calcul des classes dans les données historiques
            for col in target_columns:
                historical_data[f'Classe_{col}'] = (historical_data[col] // 50).astype('Int64')

            # Fusion des données historiques avec les prédictions en utilisant un 'inner' join pour n'afficher que les lignes communes
            merged_data_table = pd.merge(historical_data, predictions, on='Datetime', how='inner')

            # Réorganisation des colonnes pour mettre chaque prédiction à côté de sa colonne historique
            comparison_columns = ['Datetime']
            for col in target_columns:
                historical_class_col = f'Classe_{col}'
                predicted_col = f'random_forest_{col}'
                columns_to_add = []
                if historical_class_col in merged_data_table.columns:
                    columns_to_add.append(historical_class_col)
                if predicted_col in merged_data_table.columns:
                    columns_to_add.append(predicted_col)
                if columns_to_add:
                    comparison_columns.extend(columns_to_add)

            merged_data_table = merged_data_table[comparison_columns]

            st.write("Tableau comparatif : valeurs historiques (classes) et prédictions")
            st.write(merged_data_table)

        st.subheader("Comparer les prédictions et les valeurs historiques (graphes)")

        # Pour les graphiques, fusionner avec un 'outer' join pour inclure toutes les prédictions
        merged_data_plot = pd.merge(historical_data, predictions, on='Datetime', how='outer')

        prediction_dates = predictions['Datetime']
        start_pred = prediction_dates.min()
        end_pred = prediction_dates.max()
        mask = (merged_data_plot['Datetime'] >= start_pred) & (merged_data_plot['Datetime'] <= end_pred)
        merged_data_plot = merged_data_plot.loc[mask]

        with st.form(key="comparison_form"):
            comparison_motif = st.text_input("Saisissez le motif pour comparer les colonnes (exemple: 2R-P01 ou 2R) :")
            compare_button = st.form_submit_button(label="Afficher les graphes de comparaison")

        if compare_button and comparison_motif:
            comparison_columns = []
            for col in target_columns:
                if comparison_motif.lower() in col.lower():
                    historical_class_col = f'Classe_{col}'
                    predicted_col = f'random_forest_{col}'
                    if historical_class_col in merged_data_plot.columns or predicted_col in merged_data_plot.columns:
                        comparison_columns.append((historical_class_col, predicted_col))

            if comparison_columns:
                for historical_class_col, predicted_col in comparison_columns:
                    cols_to_plot = []
                    if historical_class_col in merged_data_plot.columns:
                        cols_to_plot.append(historical_class_col)
                    if predicted_col in merged_data_plot.columns:
                        cols_to_plot.append(predicted_col)

                    if cols_to_plot:
                        st.write(f"Comparaison entre {historical_class_col} (valeur réelle) et {predicted_col} (prédiction)")
                        
                        # Créer un graphique avec Matplotlib
                        fig, ax = plt.subplots(figsize=(10, 5))

                        # Tracer les données historiques et les prédictions
                        ax.plot(merged_data_plot['Datetime'], merged_data_plot[historical_class_col], label=historical_class_col, color='blue')
                        ax.plot(merged_data_plot['Datetime'], merged_data_plot[predicted_col], label=predicted_col, color='lightblue')

                        # Ajouter les labels des axes
                        ax.set_xlabel("Date", fontsize=12)
                        ax.set_ylabel("Classe du flux des véhicules", fontsize=12)
                        ax.set_title(f"Comparaison des flux : {historical_class_col} vs {predicted_col}", fontsize=14)

                        # Ajouter une légende
                        ax.legend()

                        # Rotation des dates pour qu'elles soient plus lisibles
                        plt.xticks(rotation=45)

                        # Simplifier le style, enlever les bordures inutiles
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(True)

                        ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
                        ax.xaxis.grid(False)
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                        # Afficher le graphique dans Streamlit
                        st.pyplot(fig)
            else:
                st.write("Aucune colonne ne correspond au motif pour la comparaison.")

    # if data_file:
    #     st.subheader("Comparaison des prédictions avec les valeurs historiques (tableaux)")

    #     # Sélection des colonnes cibles
    #     target_columns = [col for col in df.columns if col.startswith('VL') or col.startswith('PL') or col.startswith('2R')]
    #     historical_data = df[['Datetime'] + target_columns]

    #     # Calcul des classes dans les données historiques
    #     for col in target_columns:
    #         historical_data[f'Classe_{col}'] = (historical_data[col] // 50).astype('Int64')

    #     # Fusion des données historiques avec les prédictions en utilisant un 'outer' join pour inclure toutes les prédictions
    #     merged_data = pd.merge(historical_data, predictions, on='Datetime', how='outer')

    #     # Réorganisation des colonnes pour mettre chaque prédiction à côté de sa colonne historique
    #     comparison_columns = ['Datetime']
    #     for col in target_columns:
    #         historical_class_col = f'Classe_{col}'
    #         predicted_col = f'random_forest_{col}'
    #         columns_to_add = []
    #         if historical_class_col in merged_data.columns:
    #             columns_to_add.append(historical_class_col)
    #         if predicted_col in merged_data.columns:
    #             columns_to_add.append(predicted_col)
    #         if columns_to_add:
    #             comparison_columns.extend(columns_to_add)

    #     merged_data = merged_data[comparison_columns]

    #     st.write("Tableau comparatif : valeurs historiques (classes) et prédictions")
    #     st.write(merged_data)

    #     st.subheader("Comparer les prédictions et les valeurs historiques (graphes)")

    #     with st.form(key="comparison_form"):
    #         comparison_motif = st.text_input("Saisissez le motif pour comparer les colonnes (exemple: 2R-P01 ou 2R) :")
    #         compare_button = st.form_submit_button(label="Afficher les graphes de comparaison")

    #     if compare_button and comparison_motif:
    #         comparison_columns = []
    #         for col in target_columns:
    #             if comparison_motif.lower() in col.lower():
    #                 actual_col = f'Classe_{col}'
    #                 predicted_col = f'random_forest_{col}'
    #                 if actual_col in merged_data.columns and predicted_col in merged_data.columns:
    #                     comparison_columns.append((actual_col, predicted_col))

    #         if comparison_columns:
    #             for actual_col, predicted_col in comparison_columns:
    #                 st.write(f"Comparaison entre {actual_col} (valeur réelle) et {predicted_col} (prédiction)")
    #                 st.line_chart(merged_data.set_index('Datetime')[[actual_col, predicted_col]])
    #         else:
    #             st.write("Aucune colonne ne correspond au motif pour la comparaison.")