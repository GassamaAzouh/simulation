#!/usr/bin/env python
import sys
import streamlit as st
import argparse
import warnings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import scipy.stats as stats
from simulation.Crews.Scenario1.crew import Simulation
from simulation.Crews.Scenario2.crew import simulation



warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run(n_client, n_obs, n_month):
    """
    Run the crew.
    """ 
    inputs = {
        'n_client': n_client ,
        "n_obs_ref" : 30,
        "transaction_generator_agent" : " Generate Bank Transactions ",
        "rules_agent" : "implement detection rules on banking observations" ,
        "transaction_classifier_agent" : "Classification of banking transaction",
        "generator_agent" : "Generate card transactions using agent {transaction_generator_agent} information" ,
        "n" : n_obs ,
        "nb_month" : n_month ,
        #'nb_month' : 3
    }
    #Simulation().crew().kickoff(inputs=inputs)

def Run(n_clients, n_obs_n, n_obs_f) :
    
    input = {
        "n" : n_clients ,
        "nb" : n_obs_n ,
        "nb_fr" : n_obs_f
    }
    #simulation().crew().kickoff(inputs= input)

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Simulation().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Simulation().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Simulation().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def style(fig) :
    fig.update_traces(
        marker=dict(
            line=dict(
            color="black",  # Couleur des bordures
            width=2         # Épaisseur des bordures
            )
            ), textposition="outside"
                )
    
def conf_graph(fig, Title, x_name, y_name) :
    fig.update_traces(textfont_size=14, textposition='outside')
    fig.update_layout(            
        title=dict(
            text=Title,
            x=0.5,
            xanchor="center",
            font=dict(size=25)
        ),
        
        plot_bgcolor="rgba(0,0,0,0)"  ,
        xaxis=dict(
            title=dict(text=x_name, font=dict(size=20)),
            tickfont=dict(size=18)
        ),
        yaxis=dict(
            title=dict(text=y_name, font=dict(size=20)),  
            tickfont=dict(size=18)
        ),
        )

def Analyse_data(dat) :
    #dat['Date'] = dat['Date'].str.replace(r"(\d{2})h:", r"\1:", regex=True, )
    #dat['Date'] = pd.to_datetime(dat['Date'], format="%d/%m/%Y %H:%M")
            
    dat['Date'] = pd.to_datetime(dat['Date'], format="%d/%m/%Y %H:%M", errors='coerce')
    
# Supprimer les lignes où la conversion a échoué (NaT)
    dat = dat.dropna(subset=['Date'])
    dat = dat.dropna()
    dat = dat.sort_values(by="Date", ascending=False, ).reset_index(drop=True)
    dat = dat.drop_duplicates()
    #dat = dat.copy(ignore_index = True)
    

# Convertir en datetime
    dat['Date'] = pd.to_datetime(dat['Date'], format="%d/%m/%Y %H:%M")

    #dat.sort_values(by="Date", ascending=False, ) 
    #corr_num, corr_qual = st.columns([2, 1])
    #with corr_num :
    st.write("Les 5 premiers transactions")
    st.write(dat.head())
    #st.write(dat.tail())
    
    fig = px.histogram(
        dat, 
        x=dat['Montant'], 
        nbins=15, 
        title="Distribution des montants des transactions", 
        labels={"x": "Montant"}, 
        #marginal="box",  # Ajouter un boxplot en haut
        color_discrete_sequence=["blue"] ,
        width= 1200 ,
        height= 600
                )
    style(fig)
    conf_graph(fig, "Distribution des montants des transactions","Montant des transactions", 
               "Fréquence de transactions")
    fig.update_layout(
        xaxis=dict(tickformat=","),
    )
    st.plotly_chart(fig)

                # Comptage des occurrences par type de transaction
    operation_counts = dat["Type de transaction"].value_counts().reset_index()
    operation_counts.columns = ["Type de transaction", "Fréquence"]
    #operation_counts['text'] = operation_counts['Fréquence'].astype(str)  # Convertir les valeurs en texte
    fig = px.bar(
        operation_counts,
        x="Type de transaction",
        y = "Fréquence" ,
        title="Fréquence des transactions en fonction du type",
        labels = {"Type de transaction" : "Type de Transaction", "Fréquence": "Nombre de Transactions" },
        color="Type de transaction",
        text = "Fréquence" , # Ajouter les valeurs sur les barres
        #text = 'text',
        width= 1200 ,
        height= 600
                )
                
    style(fig)
    conf_graph(fig, "Fréquence des transactions en fonction du type", "Type de transaction",
               "Fréquence des types de transactions")
    fig.update_layout(
        legend=dict(
        font=dict(size=18)
        )
    )
    st.plotly_chart(fig)   

    col1, col2 = st.columns(2)
    with col1 :
        d = dat["Localisation"].value_counts().reset_index()
        d.columns = ["Localisation", "Nombre de Transactions"]
        top_regions = d.sort_values(by="Nombre de Transactions", ascending=False).head(5)
        # Graphique en barres horizontal
        fig = px.bar(
            top_regions,
            x="Nombre de Transactions",
            y="Localisation",
            orientation="h",  # Barres horizontales
            #title="Top 5 des lieux avec le plus de Transactions",
            labels={"Nombre de Transactions": "Nombre de Transactions", "Localisation": "Localisation"},
            text="Nombre de Transactions" , # Ajouter les valeurs sur les barres
            #width= 8
            #height= 15
        )
        style(fig) 
        conf_graph(fig, "Top 5 des lieux avec le plus de Transactions", "Nombre de Transactions", 
                    "Lieux de transaction")
        st.plotly_chart(fig)

    with col2 :
        d = dat["Localisation"].value_counts().reset_index()
        d.columns = ["Localisation", "Nombre de Transactions"]
        tail_regions = d.sort_values(by="Nombre de Transactions", ascending=True).head(5)
        # Graphique en barres horizontal
        fig = px.bar(
            tail_regions,
            x="Nombre de Transactions",
            y="Localisation",
            orientation="h",  # Barres horizontales
            #title="Top 5 des lieux avec le moins de Transactions",
            labels={"Nombre de Transactions": "Nombre de Transactions", "Localisation": "Localisation"},
            text="Nombre de Transactions" , # Ajouter les valeurs sur les barres
            #width= 8
        )  
        style(fig)
        conf_graph(fig, "Top 5 des lieux avec le moins de Transactions", "Nombre de Transactions", 
                    "Lieux de transaction")
        st.plotly_chart(fig)

    
    dat["hour"] = dat["Date"].dt.hour
    transactions_per_hour = dat.groupby("hour").size().reset_index()
    transactions_per_hour.columns = ["Heure", "Nombre de Transactions"]
    fig = px.bar(
        transactions_per_hour,
        x="Heure",
        y="Nombre de Transactions",
        #orientation="h",  
        labels={"Heure": "Heure", "Nombre de Transactions": "Nombre de Transactions"},
        text="Nombre de Transactions" , 
        width= 1200 ,
        height= 600
                )
    style(fig)
    conf_graph(fig, "Le nombre de transaction en fonction de l'heure","Heure", "Nombre de Transactions" )
    #fig.update_layout(
    #    bargap=0.3,),
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig)

    a, b = st.columns(2)
    with a :
        state_operation_counts = dat["Status operation"].value_counts().reset_index()
        state_operation_counts.columns = ["Status operation", "Fréquence"]
        fig = px.bar(
            state_operation_counts,
            x="Status operation",
            y = "Fréquence" ,
            #title="Fréquence des transactions en fonction du status de l'opération",
            labels = {"Status operation" : "Status operation", "Fréquence": "Nombre de Transactions" },
            color="Status operation",
            text = "Fréquence" , # Ajouter les valeurs sur les barres
            #width= 1200 ,
            #height= 600
        )
        style(fig)
        conf_graph(fig, "Fréquence des transactions en fonction <br>du status de l'opération", 
                   "Status de l'operation", "Nombre de transactions"),
        fig.update_layout(
            legend=dict(
            font=dict(size=20)
            ))
        st.plotly_chart(fig)
        
    with b :
        target_operation_counts = dat["Target"].value_counts().reset_index()
        target_operation_counts.columns = ["Target", "Fréquence"]
        fig = px.bar(
            target_operation_counts,
            x="Target",
            y = "Fréquence" ,
            #title="Nombre de transactions en fonction du target",
            labels = {"Target" : "Target", "Fréquence": "Nombre de Transactions" },
            color="Target",
            text = "Fréquence", # Ajouter les valeurs sur les barres
            #width= 1200 ,
            #height= 600
        )
        style(fig)
        conf_graph(fig, "Nombre de transactions en <br>fonction du target", "Target", "Nombre de transactions")
        fig.update_layout(
            legend=dict(
            font=dict(size=20)
            ))
        st.plotly_chart(fig)

    c, d = st.columns(2)
    with c :
        # Graphique croisée Status operation vs Target
        sta_targ = dat.groupby(['Status operation', 'Target']).size().reset_index(name="Nombre de transactions")

        fig = px.bar(
            sta_targ,
            x='Status operation',
            y="Nombre de transactions",
            color='Target',
            barmode='group',
            text = "Nombre de transactions",
            height=550,
            #title="Distribution croisée <br>entre le Statut de <br>l'Opération et le Type <br>de Transaction",
            color_discrete_map={
                'Normal': 'blue',
                'Suspect': 'lightblue',
                'Fraude': 'red'
            }
        )
        style(fig)
        conf_graph(fig, " Status operation VS Target ", 
                "Status operation", "Nombre de transactions")
        fig.update_layout(
            legend=dict(
            font=dict(size=20)
            ))
        st.plotly_chart(fig)

    with d :
        # Graphique croisée Status operation vs Target
        type_targ = dat.groupby(['Type de transaction', 'Target']).size().reset_index(name="Nombre de transactions")
        fig = px.bar(
            type_targ,
            x='Type de transaction',
            y="Nombre de transactions",
            color='Target',
            barmode='group',
            text = "Nombre de transactions",
            #width=1400,
            height=550,
            #title="Distribution croisée <br>entre le Statut de <br>l'Opération et le Type <br>de Transaction",
            color_discrete_map={
                'Normal': 'blue',
                'Suspect': 'lightblue',
                'Fraude': 'red'
            }
        )
        style(fig)
        conf_graph(fig, " Type de transaction VS Target ", 
                "Type de transaction", "Nombre de transactions")
        fig.update_layout(
            legend=dict(
            font=dict(size=20)
            ))
        st.plotly_chart(fig)

    
    categorical_cols = dat.select_dtypes(include=["object"]).columns
    encoder = LabelEncoder()

    for col in categorical_cols:
        dat[col] = encoder.fit_transform(dat[col])
    #with corr_qual :
    st.write("Matrice corrélation des variables")
    st.write(dat.corr())

def eval(dat) : 
     transactions = dat['Montant'].dropna()  # Supprimer les valeurs manquantes
     transactions = transactions[transactions > 1000]  # Supprimer les valeurs négatives
     #transactions = transactions[transactions < 0]  # Supprimer les valeurs négatives
     log_data = np.log(transactions)
     log_data = log_data[np.isfinite(log_data)]
     log_data = log_data[~np.isnan(log_data)]

     # Générer un Q-Q Plot pour vérifier l'adéquation à la loi de Pareto
     fig, ax = plt.subplots(figsize=(8,5))
     stats.probplot(log_data, dist="norm", plot=ax)

    # Assurer que la ligne diagonale est bien visible
     ax.get_lines()[1].set_linestyle("--")  # Convertir en pointillés
     ax.get_lines()[1].set_color("red")  # Mettre la diagonale en rouge
     ax.get_lines()[1].set_linewidth(3)  # Épaissir la diagonale

    # Ajouter un titre et afficher le plot
     ax.set_title("Q-Q Plot après transformation logarithmique", loc= "center")
     ax.set_xlabel("Quantiles théoriques (Normaux)", fontsize=12, fontweight='bold', labelpad=15, loc='center')
     ax.set_ylabel("Quantiles empiriques (Données Log-Transformées)", fontsize=12, fontweight='bold', labelpad=15, loc='center')

        # Calcul des quantiles théoriques et empiriques pour la loi de Pareto
     #quantiles_theoriques = stats.norm.ppf(np.linspace(-5, 18, len(log_data)))
     #quantiles_empiriques = np.sort(log_data)
     
# Affichage du graphique
     st.plotly_chart(fig)


def Scenario_1() :
    
    if "data_bank" not in st.session_state:
        st.session_state.data_bank = None  # Initialise avec aucune donnée

    if "dataa" not in st.session_state:
        st.session_state.dataa = None  # Initialise avec aucune donnée
    
    

    st.title("Simulation de données bancaires synthéthiques du Sénégal.")

    # Étape 1: Génération de clients et comportements
    st.header(" Simulation avec le Scénario 1 : ")
    st.write('')
    st.image("AgSc1.png", use_container_width=True)
    st.write('')

    col1, col2 = st.columns(2)
    with col1 :
        st.markdown("""
            - **Agent 1**
                - Utilise mixtral
                - Execute la tache A
                    
            - **Tache A**
                - Simulation de transactions 
                - Ses transactions serviront comme base référence

        """)

    with col2 :
        st.markdown("""
            - **Agent 2**
                - Utilise Llama
                - Execute la tache B
                    
            - **Tache B**
                - Analyse le comportement de chaque client sur la base de référence 
                - Met en place des règles pour chaque client pour identifier les futurs transactions des 
                    clients (Normal, Suspect ou Fraude)

        """)

    col3, col4 = st.columns(2)
    with col3 :
        st.markdown("""
            - **Agent 3**
                - Utilise Llama
                - Execute la tache C
                - La tache C dépend de A aussi
                    
            - **Tache C**
                - Simulations de nouvelles transactions 
                - Ses nouvelles transactions seront le résultat de la simulation après leur 
                    classification


        """)

    with col4 :
        st.markdown("""
            - **Agent 4**
                - Utilise Genimi
                - Execute la tache D
                - La tache D dépend de B et C 
                    
            - **Tache D**
                - Classification des nouvelles transactions (Normal, Suspect ou Fraude) 
                - Cette classification se base sur le résultat de l’analyse des comportements des clients 
                    sur la base de référence (tache B) et des règles (tache C) mise en place
        """)

    n_client = st.number_input("Nombre clients ", min_value=3, max_value=10, value=5)
    n_obs = st.number_input("Nombre transactions ", min_value=20, max_value=300, value=100)
    n_month = st.number_input("Nombre de mois pour la base de référence ", min_value=1, max_value=12, value=4)
#C:\Users\JJ\OneDrive\Desktop\Mem\simulation\Transactions_references.csv

    # État du premier bouton
    if "button_exec_clicked" not in st.session_state:
        st.session_state.button_exec_clicked = False  # Initialise l'état à False
        
    if st.button(" Executer "):
        st.session_state.button_exec_clicked = True  # Met à jour l'état du bouton
        try :
            st.write(" Veillez patientez la simulation peut prendre quelques minutes")
            st.write(" Simulation en cours ... ")
            result = run(n_client, n_obs, n_month)
            st.write(" Simulation terminée ! ✅ ") 
        except Exception as e :
            st.error(f" Une erreur inattendue est survenue : {e}")
            st.write(f"Il se peut que le RateLimit a été dépassé. Essayez de diminuer \
                    les valeurs de vos arguments.")
        
    if st.session_state.button_exec_clicked:
        
        left, middle, right = st.columns(3)

        dataa = pd.read_csv("Transactions_references.csv", sep = ";")

        #dataa.corr(numeric_only = True)
        #dataa = st.session_state.dataa.copy()
            #dataa.head()
        file_ref = dataa.to_csv(index=False, sep=';')

        with left :
            
            st.markdown('<div class="center-content">', unsafe_allow_html=True)
            st.download_button(
            label="Télécharger les Transactions Synthétiques de références",
            data = file_ref ,
            file_name="Transactions_references.csv",
            mime="text/csv"
            )
            st.markdown('<div class="center-content">', unsafe_allow_html=True)
        
        
        #st.session_state.data_bank = pd.read_csv("Bank_transaction.csv", sep=';', skiprows=1)
        st.session_state.data_bank = pd.read_csv("Bank_transaction.csv", sep = ";")
        dat = st.session_state.data_bank.copy()
        file_tr = dat.to_csv(index=False, sep=';')    

        with right :        
            st.markdown('<div class="center-content">', unsafe_allow_html=True)
            st.download_button(
            label="Télécharger les transactions bancaires synthétiques ",
            data = file_tr ,
            file_name="Transactions_bancaires_sénégalais.csv",
            mime="text/csv"
            )
            st.markdown('<div class="center-content">', unsafe_allow_html=True)
        file_rule = "Regles.txt"
        with middle :
            st.markdown('<div class="center-content">', unsafe_allow_html=True)
            st.download_button(
            label="Télécharger les règles des clients ",
            data = file_rule ,
            file_name="Règles.txt",
            mime="text/csv"
            )
            st.markdown('<div class="center-content">', unsafe_allow_html=True)

        #st.markdown('<div class="center-content">', unsafe_allow_html=True)

    

        if st.session_state.data_bank is not None: 
            #st.markdown('<div class="center-content">', unsafe_allow_html=True)
            dat = st.session_state.data_bank.copy()
            if st.button(" Analyse des données ") :    
                Analyse_data(dat)
            if st.button(" Evaluation des données ") :
                eval(dat)
         
    else:
        st.warning("Veuillez d'abord cliquer sur le Bouton Exécuter pour générer des données avant de passer à l'étape d'analyse.")

    

def Acceuil() :
    
    st.title("Bienvenue dans notre plateforme de simulation de données synthétiques bancaires basés sur un système multi-agent IA.")
    st.write("")
    st.write("Cette plateforme intelligente repose sur des agents IA collaboratifs utilisant des \
            modèles de langage (LLM) pour simuler des données bancaires réalistes. Le projet a pour \
            objectif de générer des transactions bancaires synthétiques. La génération se fait par le \
            biais deux Scénarios. Pour chacune des deux scénarios, les agents ont une ou des taches à \
            exécuter. Voici un petit résumé de nos deux scénarios de simulation. Pour plus de détails\
             allez dans la navigation de gauche pour voir le fonctionnement et l'architecture des scénarios. ")
    st.write("")
    col0,col1 = st.columns(2)
    with col0 :
        st.markdown("""
           - **Scénario 1**
                - L'utilisateur donne les entrées du scénario qui seront envoyés au système multi-agent
                - L'agent 1 simule des transactions qui serviront comme base de référence 
                - Ses transactions seront données à l'agent 2 pour qu'il fait l'analyse du comportement \
                    des clients et met en place des règles pour identifier les futurs transactions des \
                    clients (normal, suspect ou fraude) 
                - La base référence de l'agent 1 sera envoyée à l'agent 3. Ce dernier analyse le format \
                    des données puis simule de nouvelles transactions avec le même format 
                - Les règles mis en place par l'agent 2 et les nouvelles transactions de l'agent 3 seront \
                    données à l'agent 4 qui classifiera les transactions par le biais des règles 
                - Les résultats de la classification vont etre exploités
            
        """)
        
    with col1 :
        st.markdown("""
           - **Scénario 2**
                - L'utilisateur donne les entrées du scénario qui seront envoyés au système multi-agent
                - L'agent 1' définit des groupes ou catégories de comportements de clients, puis simule \
                    des clients. Chacun sera intégré dans un groupe  
                - L'agent 2' utilisera les informations de l'agent 1' pour simuler des transactions avec \
                    le comportement habituel des clients. Autrement dit, simuler des transactions normales  
                - L'agent 3' prend les informations de l'agent 2' et l'agent 1' pour simuler des \
                    transactions qui ne respectent pas les comportements habituels (transactions \
                    frauduleuses) des clients et avec le même format données que les transactions \
                    habituels simulés  
                - Le résultat des deux simulations (Normales et Fraude) seront combinés puis exploités 

        """)

def Scenario_2() :
    if "data_bank" not in st.session_state:
        st.session_state.data_bank = None

    st.title("Simulation de données bancaires synthéthiques du Sénégal.")
    st.header(" Simulation avec le Scénario 2 : ")
    st.image("AgSc2.png", use_container_width = True, width=150)
    st.write("")
    

    col_1,col_2, col_3 = st.columns(3)
    with col_1 :
        st.markdown("""
            - **Agent 1'**
                - Utilise Genimi
                - Exécute la tache A'
                    
            - **Tache A'**
                - Simulation de comportements de groupes de clients 
                - Simulation de clients (code client et numéro de compte)
                - Chaque client sera intégré dans un groupe
        """)
    with col_2 :
        st.markdown("""
            - **Agent 2'**
                - Utilise llama
                - Exécute la tache B'
                - La tache B' dépend de A'
                    
            - **Tache B'**
                - Simulation des transactions en se basant sur les comportements des 
                    clients qui sont définit. 
                - Ses transactions seront considérés comme **Normales** car elles respectent le comportement 
                    habituel des clients.
        """)
    with col_3 :
        st.markdown("""
            - **Agent 3'**
                - Utilise llama
                - Exécute la tache C'
                - La tache C' dépend de A' et B'
                    
            - **Tache C'**
                - Simulation des transactions qui ne respectent pas les comportements des clients qui 
                    ont été définit.
                - Ses transactions seront considérés comme **Frauduleuses** car elles respectent le comportement 
                    habituel des clients.
        """)

    n_clients = st.number_input("Nombre clients ", min_value = 1, max_value = 20, value = 5)
    n_obs = st.number_input("Nombre transactions ", min_value=50, max_value=500, value=100)
    p_fr = st.number_input("Pourcentage de transaction de fraude ", min_value=1, max_value=15, value=2)
    n_obs_f = round((n_obs * p_fr)/100)
    n_obs_n = n_obs - n_obs_f

    # État du premier bouton
    if "button_exec2_clicked" not in st.session_state:
        st.session_state.button_exec2_clicked = False  
        
    if st.button(" Executer   "):
        st.session_state.button_exec2_clicked = True  # Met à jour l'état du bouton
        try :
            st.write(" Simulation en cours ... ")
            result = Run(n_clients, n_obs_n, n_obs_f)
            st.write(" Simulation terminée ! ✅ ") 
        except Exception as e :
            st.error(f"Une erreur inattendue est survenue : {e}")
            st.write(f"Il se peut que le RateLimit a été dépassé. Si c'est le cas essayer de diminuer \
                    les valeurs de vos arguments.")
        
    nor_path = "Normales.csv"
    fr_path = "Fraude.csv"
    if st.session_state.button_exec2_clicked:

        left, right = st.columns(2)

        data_nor = pd.read_csv("Normales.csv", sep = ";")
        data_fr = pd.read_csv("Fraude.csv", sep = ";")
        data = pd.concat([data_nor, data_fr], ignore_index=True)
        
        
        st.session_state.data_bank = data
        #dataa = st.session_state.dataa.copy()
            #dataa.head()
        dataa = data.to_csv(index=False, sep=';')
        
        with left :          
            st.markdown('<div class="center-content">', unsafe_allow_html=True)
            st.download_button(
            label="Télécharger les transactions bancaires synthétiques",
            data = dataa ,
            file_name="Transactions_synthétiques.csv",
            mime="text/csv"
            )
            st.markdown('<div class="center-content">', unsafe_allow_html=True)

        client = "Clients.txt"    
        with right :
            st.markdown('<div class="center-content">', unsafe_allow_html=True)
            st.download_button(
            label="Télécharger les clients générés et leurs comportements",
            data = client ,
            file_name="Règles.txt",
            mime="text/csv"
            )
        #dat = st.session_state.data_bank.copy()

        if st.session_state.data_bank is not None: 
            dat = st.session_state.data_bank.copy()
                #st.markdown('<div class="center-content">', unsafe_allow_html=True)
            if st.button(" Analyse des données ") :    
                   
                Analyse_data(dat)
            if st.button(" Evaluation des données ") :
                eval(dat)
          
    else:
            st.warning("Veuillez d'abord cliquer sur le Bouton Exécuter pour générer des données avant de passer à l'étape d'analyse.")



if __name__ == "__main__" :

    # --- Configuration de l'application ---
    st.set_page_config(
        page_title="Simulateur de données bancaires",
        page_icon="📊",
        layout="wide"
    )

    # --- Barre latérale pour la navigation ---
    st.sidebar.title("Navigation")
    pages = ["Accueil", "Scenario 1", "Scenario 2"]
    selected_page = st.sidebar.radio("Choisissez le scénario souhaitez :", pages)

    if selected_page == "Accueil" :
        Acceuil()
    elif selected_page == "Scenario 1" :
        Scenario_1()
    elif selected_page == "Scenario 2" :
        Scenario_2()

    

    




        
