import streamlit as st
import pandas as pd
from pathlib import Path
import os
from VueGlobale import VueGlobale
from Categorisation import Categorisation
from AnalaysesSecondaires import AnalysesSecondaires
from FicheRef import FicheRef
from streamlit_option_menu import option_menu
from NouvRef import NouvRef
import locale
import datetime
st. set_page_config(layout="wide")

#GEI = Groupe d'elements interchangeables



def get_data():
    st.markdown(f'<p style="color: black; font-size: 15px ; font-weight : bold; ">Fichier des mouvements à analyser</p>',unsafe_allow_html=True)
    file_excel = st.file_uploader( "", key="mouvements")
    return file_excel

def get_stock():
    st.markdown(f'<p style="color: black; font-size: 15px ; font-weight : bold; ">Fichier du stock à analyser</p>', unsafe_allow_html=True)
    file_stock = st.file_uploader("", key="stock")
    return file_stock

def setDateDebut(dateDebut):
    if (dateDebut.month == 12) and (dateDebut.day != 1):
        date = str(dateDebut.year + 1) + "-01-01"
        dateDebut = pd.Timestamp(date)
    if (dateDebut.month !=12) and (dateDebut.day != 1):
        mois = dateDebut.month +1
        date = str(dateDebut.year) + "-" + str(mois) + "-1"
        dateDebut = pd.Timestamp(date)
    return dateDebut

def setDateFin(dateFin):
    if (dateFin.month ==1) and(dateFin.day != months[1]):
        date = str(dateFin.year - 1) + "-12-31"
        dateFin = pd.Timestamp(date)
    if ((dateFin.month == 2) and (dateFin.year % 4 == 0 ) and(dateFin.day != 29)):
        date = str(dateFin.year) + "-01-31"
        dateFin = pd.Timestamp(date)

    if ((dateFin.month == 2) and ( dateFin.year % 4 != 0 ) and (dateFin.day != 28)):
        date = str(dateFin.year) + "-01-31"
        dateFin = pd.Timestamp(date)

    if (dateFin.month != 2) and (dateFin.day != months[dateFin.month]) :
        date = str(dateFin.year) + "-" + str(dateFin.month-1) + "-" + str(months[dateFin.month -1])
        dateFin = pd.Timestamp(date)
    return dateFin




if not os.path.exists("Log"):
    os.mkdir("Log")
log_file = open("Log/Log_File.txt", "a")



try :
    Affichages = {}

    Affichages['Vue globale'] = VueGlobale()
    Affichages['Catégorisation'] = Categorisation()
    Affichages['Analyses secondaires'] = AnalysesSecondaires()
    Affichages["Catégorisation d'une nouvelle référence"] = NouvRef()
    Affichages["Fiche d'une référence"] = FicheRef()

    months = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31
    }
    COLONNES_MOUV = ['Libellé_activité', 'Code_article', 'Libellé_article', 'Code_mouvement', 'Libellé_mouvement',
                     'Date_création', 'Quantité']


    COLONNES_STOCK = ['Libellé_activité', 'Code_article', 'Libellé_article',
                      'Nom_emplacement', 'Quantité', 'Date_création']

    current_path = Path(os.path.abspath('')).resolve()


    themes = ["Accueil", "Vue globale", "Catégorisation", "Analyses secondaires", "Catégorisation d'une nouvelle référence","Fiche d'une référence"]
    icons = ["house", "gear", "list-task", "", "", "",""]

    IndiceMouvements = ["Code Mouvement de GEI", "VG_Code type de mouvement de GEI", "Code mouvement de GEI","VG_Code type de mouvement de GEI","Code mouvement"]


    with st.sidebar:
        selected_theme = option_menu("IA - MAGASIN", themes, icons=icons, menu_icon="cast", default_index=0, styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#0F056B", "font-size": "25px"},
                "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#eee","font-weight" : "bold","color" : "#0F056B"},
                "nav-link-selected": {"background-color": "#ADD8E6"}
            })



    # Variable mouvementHint est une variable binaire :
    # mouvementHint = 0 ==> il s'agit d'un fichier du stock
    # mouvementHint = 1 ==> il s'agit d'un fichier des mouvements

    mouvementHint = 0
    col1,col2 = st.columns(2)
    with col1 :
        file_excel = get_data()


        if file_excel is not None:
            xl_file = pd.read_excel(file_excel, engine='openpyxl')
            for indice in IndiceMouvements:
                if indice in xl_file.columns.tolist():
                    mouvementHint += 1
                else:
                    mouvementHint += 0

            if mouvementHint > 0:
                xl_file.columns = COLONNES_MOUV
                xl_file['Libellé_activité'] = xl_file['Libellé_activité'].astype(str)
                xl_file['Libellé_article'] = xl_file['Libellé_article'].astype(str)
                xl_file['Code_article'] = xl_file['Code_article'].astype(str)
                xl_file['Code_mouvement'] = xl_file['Code_mouvement'].astype(str)
                xl_file['Libellé_mouvement'] = xl_file['Libellé_mouvement'].astype(str)

                ficheRefDf = xl_file.copy()
                endDate = xl_file.Date_création.max()
                startDate = xl_file.Date_création.min()

                startDate = setDateDebut(startDate)
                endDate = setDateFin(endDate)

                xl_file = xl_file[xl_file["Date_création"].isin(
                    pd.date_range(start=startDate,
                                  end=endDate))]


                locale.setlocale(locale.LC_ALL,"fr_FR")
                listeJour = []
                xl_file["Nbre de mvt Par Jour"] = 0
                for i in xl_file.index:

                    listeJour.append(datetime.datetime.strftime(xl_file["Date_création"][i], "%a"))
                    xl_file["Nbre de mvt Par Jour"][i] = xl_file[xl_file["Date_création"] == xl_file["Date_création"][i]].shape[0]
                xl_file["jour"] = listeJour




                with st.expander("Quel format de fichier dois-je utiliser ? "):
                    st.write("""
                            * 
                                Vous devez placer les colonnes dans cet ordre:
                                - Libellé activité	
                                - Code article	
                                - Libellé article	
                                - Code mouvement de GEI	
                                - Libelle mouvement de GEI	
                                - Date du mouvement	
                                - Quantite de base	
                                
                                Les fichiers doivent être au format Excel (.xlsx)
                
                                Vous devez supprimer:
                                - les pages ne correspondant pas à un magasin
                                - Les colonnes et les lignes avant le tableau
                                (les données doivent commencer en A1)
                
                                Les pages invisibles sont ignorées.
                
                                Les pages correspondant à un même magasin doivent avoir le même nom. 
              
                            """)
    with col2:
        fichier_stock = get_stock()
        if fichier_stock is not None:
            stock_file = pd.read_excel(fichier_stock, engine='openpyxl')
            stock_file.columns = COLONNES_STOCK


            with st.expander("Quel format de fichier dois-je utiliser ? "):
                st.write("""
        
                        * 
                            Vous devez placer les colonnes dans cet ordre:
                            - Libellé activité	
                            - Code article	
                            - Libellé article	
                            - Nom Emplacement	
                            - Quantité de base	
                            - Date de réception 
                           \n\n
                            
                        *   Les fichiers doivent être au format Excel (.xlsx)
                            
                            Vous devez supprimer:
                            - les pages ne correspondant pas à un magasin
                            - Les colonnes et les lignes avant le tableau
                            (les données doivent commencer en A1)
            
                            Les pages invisibles sont ignorées.
            
                            Les pages correspondant à un même magasin doivent avoir le même nom. 
               
                        """)








    #region Tableau des analyses disponibles en fonction des fichiers fournis
    listeAnalyse = ["Vue Globale","Catégorisation","Catégorisation d'une nouvelle référence",
                     "Erreur de l'inventaire","Articles en potentielle rupture","Stock mort","Fiche d'une reference"]


    if(file_excel is None) and (fichier_stock is None):
        listeDisponibilte = ["❌","❌","❌","❌","❌","❌","❌"]
        AnalysesDisponiblesDf = pd.DataFrame(list(zip(listeAnalyse, listeDisponibilte)),
                                             columns=["Analyse", 'Disponible']).transpose()

        st.table(AnalysesDisponiblesDf.style.set_table_styles([{'selector': 'th',
                                                                "props": [
                                                                    ("color", "black"),
                                                                    ("font-weight", "bold"),
                                                                    ("font-size", "20px")
                                                                ]},
                                                               {"selector": "th.row_heading",
                                                                "props": [
                                                                    ("color", "black"),
                                                                    ("font-weight", "bold"),
                                                                    ("font-size", "18px"),
                                                                    ("text-align", "center")
                                                                ]
                                                                },
                                                               {"selector" : "td",
                                                                "props": [
                                                                    ("font-size", "18px")
                                                                ]
                                                                }
                                                               ])

                 )
    if (file_excel is not None) and (xl_file.shape[0] > 200) and fichier_stock is None :
        listeDisponibilite = ["✔️", "✔️", "✔️", "✔️", "❌", "❌", "✔️"]
        AnalysesDisponiblesDf = pd.DataFrame(list(zip(listeAnalyse, listeDisponibilite)),
                                             columns=["Analyse", 'Disponible']).transpose()

        st.table(AnalysesDisponiblesDf.style.set_table_styles([{'selector': 'th',
                                            "props": [
                                                ("color", "black"),
                                                ("font-weight", "bold"),
                                                ("font-size", "20px")
                                            ]},
                                                               {"selector" :"th.row_heading",
                                                                "props": [
                                                                    ("color", "black"),
                                                                    ("font-weight", "bold"),
                                                                    ("font-size", "18px"),
                                                                    ("text-align","center")
                                                                ]
                                                                },
                                                               {"selector" : "td",
                                                                "props": [
                                                                    ("font-size", "18px")
                                                                ]
                                                                }
                                                ])


                 )



    if (file_excel is not None) and (xl_file.shape[0] > 200) and (fichier_stock is not  None) and (stock_file.shape[0] > 200 ):
        listeDisponibilite = ["✔️","✔️","✔️","✔️","✔️","✔️","✔️"]
        AnalysesDisponiblesDf = pd.DataFrame(list(zip(listeAnalyse, listeDisponibilite)),
                                             columns=["Analyse", 'Disponible']).transpose()

        st.table(AnalysesDisponiblesDf.style.set_table_styles([{'selector': 'th',
                                                                "props": [
                                                                    ("color", "black"),
                                                                    ("font-weight", "bold"),
                                                                    ("font-size", "20px")
                                                                ]},
                                                               {"selector": "th.row_heading",
                                                                "props": [
                                                                    ("color", "black"),
                                                                    ("font-weight", "bold"),
                                                                    ("font-size", "18px"),
                                                                    ("text-align", "center")
                                                                ]
                                                                },
                                                               {"selector" : "td",
                                                                "props": [
                                                                    ("font-size", "18px")
                                                                ]
                                                                }
                                                               ])

                 )


    if (file_excel is None) and (fichier_stock is not None) and (stock_file.shape[0]>200) :
        listeDisponibilite = ["❌","❌","❌","❌","❌","❌","❌"]
        AnalysesDisponiblesDf = pd.DataFrame(list(zip(listeAnalyse, listeDisponibilite)),
                                             columns=["Analyse", 'Disponible']).transpose()

        st.table(AnalysesDisponiblesDf.style.set_table_styles([{'selector': 'th',
                                                                "props": [
                                                                    ("color", "black"),
                                                                    ("font-weight", "bold"),
                                                                    ("font-size", "20px")
                                                                ]},
                                                               {"selector": "th.row_heading",
                                                                "props": [
                                                                    ("color", "black"),
                                                                    ("font-weight", "bold"),
                                                                    ("font-size", "18px"),
                                                                    ("text-align", "center")
                                                                ]
                                                                },
                                                               {"selector" : "td",
                                                                "props": [
                                                                    ("font-size", "18px")
                                                                ]
                                                                }
                                                               ])

                 )


        #endregion







    if (selected_theme != "Accueil") and (selected_theme != "Analyses secondaires") and(selected_theme != "Fiche d'une référence") and (file_excel is not None):
        Affichages[selected_theme](xl_file)
    if selected_theme == "Analyses secondaires" :
        if (fichier_stock is not None) and (file_excel is not None):
            Affichages[selected_theme](xl_file, stock_file)
        elif file_excel is not None :
            stock_file = None
            Affichages[selected_theme](xl_file, stock_file)
        else :
            xl_file = None
            stock_file = None
            Affichages[selected_theme](xl_file, stock_file)

    if selected_theme == "Fiche d'une référence":
        if (fichier_stock is not None) and (file_excel is not None):
            Affichages[selected_theme](ficheRefDf, stock_file)
        elif file_excel is not None:
            stock_file = None
            Affichages[selected_theme](ficheRefDf, stock_file)
        else:
            stock_file = None
            ficheRefDf = None
            Affichages[selected_theme](ficheRefDf, stock_file)

except Exception as e:
    st.error("Analyse non disponible ! ")
    log_file.write(str(e))
    log_file.write("\n")
    log_file.write("\n")
    log_file.write("\n ------------------------------------------------------------------------------ ")
    log_file.write("\n")

