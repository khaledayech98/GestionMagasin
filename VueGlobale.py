import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import locale
import plotly.express as px
from io import BytesIO

#region Fonctions

@st.cache()
def trouver_abcisse(ordonne, tabx, taby):
    for i in range(len(tabx)):
        if taby[i] > ordonne:
            return (round(tabx[i-1], 2))
def impacte_type_mouvement(flux):
    """
    Etant donné le flux d'un magasin, donne les quantités cumullées
    pour chaque type de mouvement au cours du temps.
    """


    dfs = []
    types_mvt = flux.Libellé_mouvement.value_counts().index
    for type_mvt in types_mvt:
        df = flux[flux['Libellé_mouvement'] == type_mvt]
        df = df.groupby(df.index).sum()
        dfs.append(df.Quantité)

    flux_acti = pd.concat(dfs, axis=1)
    flux_acti.columns = types_mvt
    flux_acti.fillna(0, inplace=True)
    flux_acti = flux_acti.astype(int)

    flux_acti.index = pd.to_datetime(flux_acti.index)
    return flux_acti 
def hist_nombre_mouvement(flux):
    dfs = []
    types_mvt = flux.Libellé_mouvement.value_counts().index
    for type_mvt in types_mvt:
        df = flux[flux['Libellé_mouvement'] == type_mvt]
        df = df.groupby(df.index).count()
        dfs.append(df.Quantité)

    flux_acti = pd.concat(dfs, axis=1)
    flux_acti.columns = types_mvt
    flux_acti.fillna(0, inplace=True)
    flux_acti = flux_acti.astype(int)

    flux_acti.index = pd.to_datetime(flux_acti.index)
    return flux_acti
def hist_nombre_mouvement_par_jour(flux):
    dfs = []
    jours = flux.jour.value_counts().index
    for jour in jours:
        df = flux[flux['jour'] == jour]
        df = df.groupby(df.index).count()
        dfs.append(df["Nbre de mvt Par Jour"])
    flux_acti = pd.concat(dfs, axis=1)
    flux_acti.columns = jours
    flux_acti.fillna(0, inplace=True)
    flux_acti = flux_acti.astype(int)

    flux_acti.index = pd.to_datetime(flux_acti.index)
    flux_acti = flux_acti[["lun.","mar.","mer.","jeu.","ven."]]
    return flux_acti
def hist_quantite_par_jour(flux):
    dfs = []
    jours = flux.jour.value_counts().index
    for jour in jours:
        df = flux[flux['jour'] == jour]
        df["Quantité"] = abs(df["Quantité"])
        df = df.groupby(df.index).sum()
        dfs.append(df["Quantité"])
    flux_acti = pd.concat(dfs, axis=1)
    flux_acti.columns = jours
    flux_acti.fillna(0, inplace=True)
    flux_acti = flux_acti.astype(int)

    flux_acti.index = pd.to_datetime(flux_acti.index)
    flux_acti = flux_acti[["lun.","mar.","mer.","jeu.","ven."]]
    return flux_acti
def setDateDebutWithMonth(nombre_mois, datefin):
    difference = datefin.month - nombre_mois + 1
    if (difference > 0):
        date = str(datefin.year) + "-" + str(difference) + "-01"
        dateDebut = pd.Timestamp(date)

    if (difference <= 0) and (difference > -12):
        date = str(datefin.year - 1) + "-" + str(difference + 12) + "-01"
        dateDebut = pd.Timestamp(date)

    if ((difference <= -12)):
        mois = (difference - 1) % 12
        annee = datefin.year - abs((difference // 12))
        date = str(annee) + "-" + str(mois + 1) + "-01"
        dateDebut = pd.Timestamp(date)

    return dateDebut
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data
#endregion



class VueGlobale():
    def __init__(self):

        return

    def __call__(self, Mouvements):
        locale.setlocale(locale.LC_ALL, "fr_FR")
        Mouvements.index = pd.to_datetime( Mouvements.Date_création,  format='%d%m%Y')

        flux_acti = impacte_type_mouvement(Mouvements)
        hist_nb_mvt = hist_nombre_mouvement(Mouvements)
        mvt_hist_jour = Mouvements[Mouvements["Date_création"].isin(pd.date_range(start=(setDateDebutWithMonth(6,Mouvements.Date_création.max())),
                                                                                   end=Mouvements.Date_création.max()))]



        NombreArticleTotal = len(Mouvements.Code_article.unique().tolist())


        #region Informations générales sur la base de données fournie
        st.header("Vue globale de la base de données")


        e= "%d-%m-%Y"
        dateMin = datetime.datetime.strftime(Mouvements.Date_création.min(), e)
        dateMax = datetime.datetime.strftime(Mouvements.Date_création.max(), e)

        nbArticle = "{:,}".format(NombreArticleTotal).replace(',', ' ')

        st.markdown("""
            <style>
                .markdown-font { font-size : 20px }
            </style>
        """,unsafe_allow_html=True)

        infoGenerale = """<p class = "markdown-font">""" +"Base de données du <b>" + str(dateMin) + "</b> au <b>" +str(dateMax)+"</b> comprenant <b>" + str(nbArticle) + "</b> articles."+ "</p>"
        st.markdown(infoGenerale, unsafe_allow_html=True)

        #endregion


        #region Tableau récap



        NombreMvt = Mouvements[['Code_mouvement', 'Libellé_mouvement']].value_counts().sort_index()

        MvtQuantite = Mouvements.groupby(["Code_mouvement", 'Libellé_mouvement']).sum()["Quantité"].sort_index()
        listeCodemouvment = []
        listeNombreMvt = []
        listeLibelleMvt = []
        listeMvtQuantite = []
        for item in NombreMvt.items():
            listeCodemouvment.append(item[0][0])
            listeNombreMvt.append(item[1])
            listeLibelleMvt.append(item[0][1])

        for item in MvtQuantite.items():
            listeMvtQuantite.append(item[1])

        dataframeMouvement = pd.DataFrame(
            list(zip(listeCodemouvment, listeLibelleMvt, listeNombreMvt, listeMvtQuantite)),
            columns=["Code Mouvement", "Libellé Mouvement", "Nombre de mouvement", "Quantités gérées par les mouvements"])
        dataframeMouvement = dataframeMouvement.sort_values(by=["Nombre de mouvement"], ascending=False)
        dataframeMouvement["Quantités gérées par les mouvements"] = abs(dataframeMouvement["Quantités gérées par les mouvements"] )
        for index in dataframeMouvement.index:
            dataframeMouvement["Nombre de mouvement"][index] = "{:,}".format( dataframeMouvement["Nombre de mouvement"][index]).replace(',', ' ')
            dataframeMouvement["Quantités gérées par les mouvements"][index] = "{:,}".format( dataframeMouvement["Quantités gérées par les mouvements"][index]).replace(',', ' ')
        listeLibelleMvt = dataframeMouvement["Libellé Mouvement"].tolist()
        df = dataframeMouvement.set_index("Code Mouvement").transpose()
        st.table(df)

                 # .style.set_table_styles([{'selector': 'th',
                 #                        "props": [
                 #                            ("color", "black"),
                 #                            ("font-weight", "bold"),
                 #                            ("font-size", "20px")
                 #                        ]},
                 #                                           {"selector" : "td",
                 #                                            "props": [
                 #                                                ("font-size", "18px")
                 #                                            ]
                 #                                            }
                 #                            ]))

        #endregion


        #region Diagramme des types et volumes des mouvements
        st.header("Analyse par nombre de mouvements")

        with st.expander("Explication"):
            st.markdown("""
                                <style>
                                    .markdown {font-size : 15px; font-weight : bold; }      
                                </style>
                            """, unsafe_allow_html=True)
            st.markdown("""<p class = markdown>
                        Il s'agit du nombre de nombre de mouvement pour chaque mois. </p>
                    """,unsafe_allow_html=True)
        st.text("En nombre de mouvement")

        st.bar_chart(hist_nb_mvt.groupby(pd.Grouper(freq="M")).sum())





        with st.expander("ANALYSE PAR VOLUME DE QUANTITES GEREES PAR LES MOUVEMENTS",expanded=False):
            st.markdown("""
                    <style>
                        .markdown {font-size : 15px; font-weight : bold; }      
                    </style>
                """, unsafe_allow_html=True)
            st.markdown("""<p class = markdown>
                    Il s'agit du volume de quantité gérées par les mouvements pour chaque mois.</p>
            """,unsafe_allow_html=True)
            st.text("En nombre de quantité")
            st.bar_chart(flux_acti.groupby(pd.Grouper(freq="M")).sum())





        #endregion


        #region Mouvement/jour de la semaine

        st.header(" Analyse des mouvements par jour de la semaine")
        libelleChoisi = st.selectbox("Choisir un libellé de mouvement", listeLibelleMvt,key = "1")
        jours = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam."]
        mouvements = Mouvements.copy()

        optionDate = st.selectbox("Choisir un intervalle de mois", [1,3,6,9,12],index=2)

        startDate = setDateDebutWithMonth(optionDate,mouvements.Date_création.max())
        endDate = mouvements.Date_création.max()

        Mouvs = mouvements[mouvements["Date_création"].isin(pd.date_range(start=startDate, end=endDate))]


        jourdf = Mouvs[Mouvs.Libellé_mouvement == libelleChoisi]["jour"].value_counts().to_frame()

        jourdf = jourdf.rename(columns={"jour": "Nombre de mouvement"})
        jourdf["jour"] = jourdf.index
        for jour in jours:
            if jour not in jourdf["jour"].tolist():
                jourdf = jourdf.append({"Nombre de mouvement": 0, "jour": jour}, ignore_index=True)
        jourdf.index = jourdf.jour
        jourdf = jourdf.loc[jours]
        jourdf = jourdf[jourdf["Nombre de mouvement"] != 0]
        HistoJour = px.bar(jourdf, x="jour", y="Nombre de mouvement")
        HistoJour.update_xaxes(title_font = dict(size = 20),tickfont = dict(color="crimson",size=15))
        HistoJour.update_yaxes(title_font = {"size":20},tickfont=dict(color="crimson", size=15))
        st.plotly_chart(HistoJour, use_container_width=True)


        #endregion


        #region Frequence jour par jour ( avec possibilité de cocher/décocher les jours qu'on veut voir )


        try :
            st.markdown("### Détails des mouvements par jour de la semaine pour les 6 derniers mois")
            hist_nbmvt_per_jour = hist_nombre_mouvement_par_jour(mvt_hist_jour[mvt_hist_jour["Libellé_mouvement"]==libelleChoisi])
            fig = px.bar(hist_nbmvt_per_jour.groupby(pd.Grouper(freq="d")).sum())
            fig.update_layout(plot_bgcolor="rgb(255,255,255)")
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(189, 195, 199,1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(189, 195, 199,1)')
            fig.update_yaxes(title="Nombre de mouvement")
            fig.update_yaxes(title_font = {"size":20},tickfont=dict(color="crimson", size=15))
            fig.update_xaxes(title=" ")

            st.plotly_chart(fig, use_container_width=True)


            with st.expander("Détails des quantités par jour de la semaine pour les 6 derniers mois"):


                hist_qte_per_jour = hist_quantite_par_jour(mvt_hist_jour[mvt_hist_jour["Libellé_mouvement"]==libelleChoisi])
                fig = px.bar(hist_qte_per_jour.groupby(pd.Grouper(freq="d")).sum())
                fig.update_layout(plot_bgcolor="rgb(255,255,255)")
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(189, 195, 199,1)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(189, 195, 199,1)')
                fig.update_yaxes(title="Quantité")
                fig.update_yaxes(title_font={"size": 20}, tickfont=dict(color="crimson", size=15))
                fig.update_xaxes(title=" ")
                st.plotly_chart(fig, use_container_width=True)
        except Exception :
            st.error("Analyse non disponible !")

        #endregion


        #region Détail de chaque mouvement
        st.header("Répartition des mouvements en fonction des articles")

        endDate = Mouvements.Date_création.max()

        listeMois = [1,3,6,12]
        optionLibelleMouvement = st.selectbox("Choisir un libellé de mouvement", listeLibelleMvt,key="2")

        col1, col2 = st.columns(2)

        with col1:
            st.selectbox("Choisir un intervalle de mois", [3])
            dataFramePerPeriode = Mouvements[ Mouvements['Date_création'].isin(pd.date_range(start=setDateDebutWithMonth( 3, Mouvements.Date_création.max()), end=endDate))]

            #region  detail de mouvements
            dfParticuliere = dataFramePerPeriode[dataFramePerPeriode["Libellé_mouvement"] == str(optionLibelleMouvement)]
            dfParticuliere["Quantité"] = abs(dfParticuliere["Quantité"])
            NombreMvt = str(dfParticuliere.shape[0])
            NombreMvt = "{:,}".format(int(NombreMvt)).replace(',', ' ')

            NombreArticleAvecMvt = str(len(dfParticuliere.Code_article.unique().tolist()))
            NombreArticleAvecMvt = "{:,}".format(int(NombreArticleAvecMvt)).replace(',', ' ')

            NombreArticleSansMvt = str(NombreArticleTotal - len(dfParticuliere.Code_article.unique().tolist()))
            NombreArticleSansMvt = "{:,}".format(int(NombreArticleSansMvt)).replace(',', ' ')

            Quantite = str(int(dfParticuliere.Quantité.sum()))
            Quantite = "{:,}".format(int(Quantite)).replace(',', ' ')

            dictionnaire = {
                "Nombre de mouvements": [NombreMvt],
                "Quantités": [Quantite],
                "Articles avec mouvements": [NombreArticleAvecMvt],
                "Articles sans mouvement": [NombreArticleSansMvt]
            }

            dfMvtPerPeriode = pd.DataFrame(dictionnaire)
            dfMvtPerPeriode = dfMvtPerPeriode.transpose()

            st.table(dfMvtPerPeriode.style.set_table_styles([{'selector': 'th',
                                    "props": [
                                        ("color", "black"),
                                        ("font-weight", "bold"),
                                        ("font-size", "20px")
                                    ]},
                                                           {"selector" : "td",
                                                            "props": [
                                                                ("font-size", "18px")
                                                            ]
                                                            }]).hide_columns())


            #endregion
            RepartitionArticle = round(dfParticuliere[["Code_article","Libellé_article"]].value_counts() * 100 / dfParticuliere.shape[0], 2)

            listey = []
            somme = 0

            for item in RepartitionArticle.items():
                somme += item[1]
                listey.append(somme)


            tabx = []
            for i in range(1, len(listey) + 1):
                tabx.append(round(((i / len(listey)) * 100), 2))

            CodeArticles = []
            LibelleArticles = []
            for i in RepartitionArticle.index:
                CodeArticles.append(i[0])
                LibelleArticles.append(i[1])

            dfExtract = pd.DataFrame(list(zip(CodeArticles, LibelleArticles, tabx, listey)),
                                     columns=["Code_article", "Libellé_article", "Pourcentage d'article",
                                              "Pourcentage de mouvement"])

            dfExtract["Pourcentage de mouvement"] = dfExtract["Pourcentage de mouvement"].astype(str)
            dfExtract["Pourcentage d'article"] = dfExtract["Pourcentage d'article"].astype(str)
            dfExtract["Pourcentage de mouvement"] = dfExtract["Pourcentage de mouvement"] + " %"
            dfExtract["Pourcentage d'article"] = dfExtract["Pourcentage d'article"] + " %"








            tabx.append(0)
            tabx.sort()
            listey.append(0)
            listey.sort()
            taby = np.array(listey)
            abcisse80pourcent = trouver_abcisse(80, tabx, taby)
            tab20x = np.array([20, 20, 20])
            tab20y = np.array([0, 40, 100])

            tab20xx = np.array([abcisse80pourcent, abcisse80pourcent, abcisse80pourcent])
            tab20yy = np.array([0, 40, 100])

            figureMouvement1mois = plt.figure(figsize=(7,3))
            plt.plot(tabx, taby, color="black", linewidth=3.0)




            plt.grid(True)
            plt.xticks(np.arange(0, 110, step=10))
            plt.yticks(np.arange(0, 110, step=10))
            plt.plot(tab20x, tab20y, color="red", linewidth=2.0, linestyle='--')
            plt.plot(tab20xx, tab20yy, color="blue", linewidth=2.0, linestyle='--')
            plt.ylim((0, 110))
            plt.xlim((0, 110))


            plt.xlabel("Articles ( % )")
            plt.ylabel("Mouvements ( % )")

            st.pyplot(figureMouvement1mois)



            #region Tableau Indicateur

            TableauIndicateurs = pd.DataFrame(
                [[trouver_abcisse(1, taby, tabx), trouver_abcisse(20, taby, tabx),
                 trouver_abcisse(80, tabx, taby)]],
                columns=["1% des articles", "20% des articles", "80% des mouvements"])



            st.table(TableauIndicateurs.transpose().style.format('{:.2f}%').set_table_styles([{'selector': 'th',
                                    "props": [
                                        ("color", "black"),
                                        ("font-weight", "bold"),
                                        ("font-size", "20px")
                                    ]},
                                                           {"selector" : "td",
                                                            "props": [
                                                                ("font-size", "18px")
                                                            ]
                                                            }]).hide_columns())

            save_path = "MouvementPerArticle " + str(3) + " Mois.xlsx"
            df_xlsx = to_excel(dfExtract)
            st.download_button(label='📥 Télécharger ce résultat',
                               data=df_xlsx,
                               file_name=save_path)

            #endregion










        with col2:
            optionDate = st.selectbox("Choisir un intervalle de mois", listeMois,index=2)
            startDate =  setDateDebutWithMonth(optionDate,Mouvements.Date_création.max())

            dataFramePerPeriode = Mouvements[Mouvements['Date_création'].isin(pd.date_range(start=startDate,end=endDate))]


            dfParticulieres = dataFramePerPeriode[dataFramePerPeriode["Libellé_mouvement"] == str(optionLibelleMouvement)]
            dfParticulieres["Quantité"] = abs(dfParticulieres["Quantité"])
            NombreMvt = str(dfParticulieres.shape[0])
            NombreMvt = "{:,}".format(int(NombreMvt)).replace(',', ' ')



            NombreArticleAvecMvt = str(len(dfParticulieres.Code_article.unique().tolist()))
            NombreArticleAvecMvt = "{:,}".format(int(NombreArticleAvecMvt)).replace(',', ' ')

            NombreArticleSansMvt = str(NombreArticleTotal - len(dfParticulieres.Code_article.unique().tolist()))
            NombreArticleSansMvt = "{:,}".format(int(NombreArticleSansMvt)).replace(',', ' ')

            Quantite = str(int(dfParticulieres.Quantité.sum()))
            Quantite = "{:,}".format(int(Quantite)).replace(',', ' ')

            dictionnaire = {
                "Nombre de mouvements" : [NombreMvt],
                "Quantités" : [Quantite],
                "Articles avec mouvements" : [NombreArticleAvecMvt],
                "Articles sans mouvement" : [NombreArticleSansMvt]
            }

            dfMvtPerPeriode = pd.DataFrame(dictionnaire)


            st.table(dfMvtPerPeriode.transpose().style.set_table_styles([{'selector': 'th',
                                "props": [
                                    ("color", "black"),
                                    ("font-weight", "bold"),
                                    ("font-size", "20px")
                                ]},
                                                       {"selector" : "td",
                                                        "props": [
                                                            ("font-size", "18px")
                                                        ]
                                                        }]))

            RepartitionArticle = round(dfParticulieres[["Code_article","Libellé_article"]].value_counts() * 100 / dfParticulieres.shape[0], 2)

            listey = []
            somme = 0

            for item in RepartitionArticle.items():
                somme += item[1]
                listey.append(somme)

            tabx = []
            for i in range(1, len(listey) + 1):
                tabx.append(round(((i / len(listey)) * 100), 2))

            CodeArticles = []
            LibelleArticles = []
            for i in RepartitionArticle.index:
                CodeArticles.append(i[0])
                LibelleArticles.append(i[1])

            dfExtraction = pd.DataFrame(list(zip(CodeArticles, LibelleArticles, tabx, listey)),
                                     columns=["Code_article", "Libellé_article", "Pourcentage d'article",
                                              "Pourcentage de mouvement"])

            dfExtraction["Pourcentage de mouvement"] = dfExtraction["Pourcentage de mouvement"].astype(str)
            dfExtraction["Pourcentage d'article"] = dfExtraction["Pourcentage d'article"].astype(str)
            dfExtraction["Pourcentage de mouvement"] = dfExtraction["Pourcentage de mouvement"] + " %"
            dfExtraction["Pourcentage d'article"] = dfExtraction["Pourcentage d'article"] + " %"



            tabx.append(0)
            tabx.sort()
            listey.append(0)
            listey.sort()
            taby = np.array(listey)


            abcisse80pourcent = trouver_abcisse(80, tabx, taby)
            tab20x = np.array([20, 20, 20])
            tab20y = np.array([0, 40, 100])

            tab20xx = np.array([abcisse80pourcent, abcisse80pourcent, abcisse80pourcent])
            tab20yy = np.array([0, 40, 100])

            figureMouvementPerPeriode = plt.figure(figsize=(7, 3))
            plt.plot(tabx, taby, color="black", linewidth=3.0)

            plt.grid(True)
            plt.xticks(np.arange(0, 110, step=10))
            plt.yticks(np.arange(0, 110, step=10))
            plt.plot(tab20x, tab20y, color="red", linewidth=2.0, linestyle='--')
            plt.plot(tab20xx, tab20yy, color="blue", linewidth=2.0, linestyle='--')
            plt.ylim((0, 110))
            plt.xlim((0, 110))

            plt.xlabel("Articles ( % )")
            plt.ylabel("Mouvements ( % )")

            st.pyplot(figureMouvementPerPeriode)
            TableauIndicateurs = pd.DataFrame([[trouver_abcisse(1, taby, tabx),trouver_abcisse(20, taby, tabx),trouver_abcisse(80, tabx, taby)]],                                                  columns=["1% des articles", "20% des articles", "80% des mouvements"])


            st.table(TableauIndicateurs.transpose().style.format('{:.2f}%').set_table_styles([{'selector': 'th',
                                "props": [
                                    ("color", "black"),
                                    ("font-weight", "bold"),
                                    ("font-size", "20px")
                                ]},
                                                       {"selector" : "td",
                                                        "props": [
                                                            ("font-size", "18px")
                                                        ]
                                                        }]).hide_columns())

            save_path = "MouvementPerArticle " + str(optionDate) + " Mois.xlsx"
            dfextract_xlsx = to_excel(dfExtraction)
            st.download_button(label='📥 Télécharger ce résultat',
                               data=dfextract_xlsx,
                               file_name=save_path)






        with st.expander("Voir la répartition des volumes selon les populations d'article", expanded=False):
            qte3mois, qteperiode = st.columns(2)
            # region  f(Article) = Quantite
            with qte3mois :
                try:
                    ArticlePerQuantite = round(
                        (dfParticuliere.groupby(
                            by=["Code_article","Libellé_article"]).Quantité.sum() * 100 / dfParticuliere.Quantité.sum()),
                        2).sort_values(ascending=False)

                    listeQuantite = []
                    SommeQuantite = 0

                    for item in ArticlePerQuantite.items():
                        SommeQuantite += item[1]
                        listeQuantite.append(SommeQuantite)
                    tabQuantiteX = []
                    for i in range(1, len(listeQuantite) + 1):
                        tabQuantiteX.append(round(((i / len(listeQuantite)) * 100), 2))

                    CodeArticles = []
                    LibelleArticles = []
                    for i in RepartitionArticle.index:
                        CodeArticles.append(i[0])
                        LibelleArticles.append(i[1])

                    dfExtractQte3 = pd.DataFrame(list(zip(CodeArticles, LibelleArticles, tabQuantiteX, listeQuantite)),
                                             columns=["Code_article", "Libellé_article", "Pourcentage d'article",
                                                      "Pourcentage de mouvement"])

                    dfExtractQte3["Pourcentage de mouvement"] = dfExtractQte3["Pourcentage de mouvement"].astype(str)
                    dfExtractQte3["Pourcentage d'article"] = dfExtractQte3["Pourcentage d'article"].astype(str)
                    dfExtractQte3["Pourcentage de mouvement"] = dfExtractQte3["Pourcentage de mouvement"] + " %"
                    dfExtractQte3["Pourcentage d'article"] = dfExtractQte3["Pourcentage d'article"] + " %"

                    listeQuantite.append(0)
                    tabQuantiteX.append(0)

                    listeQuantite.sort()
                    tabQuantiteX.sort()

                    tabQuantiteY = np.array(listeQuantite)


                    abcisse80Quantite = trouver_abcisse(80, tabQuantiteX, tabQuantiteY)
                    tab20QuantiteX = np.array([20, 20, 20])
                    tab20QuantiteY = np.array([0, 40, 100])

                    tab20Quantitexx = np.array([abcisse80Quantite, abcisse80Quantite, abcisse80Quantite])
                    tab20Quantiteyy = np.array([0, 40, 100])

                    figureQuantite1Mois = plt.figure(figsize=(7, 3))
                    plt.plot(tabQuantiteX, tabQuantiteY, color="black", linewidth=3.0)

                    plt.grid(True)
                    plt.xticks(np.arange(0, 110, step=10))
                    plt.yticks(np.arange(0, 110, step=10))
                    plt.plot(tab20QuantiteX, tab20QuantiteY, color="red", linewidth=2.0, linestyle='--')
                    plt.plot(tab20Quantitexx, tab20Quantiteyy, color="blue", linewidth=2.0, linestyle='--')
                    plt.ylim((0, 110))
                    plt.xlim((0, 110))

                    plt.plot()
                    plt.xlabel("Articles ( % )")
                    plt.ylabel("Quantités ( % )")

                    st.pyplot(figureQuantite1Mois)

                    # region Tableau Indicateurs

                    TableauIndicateurs = pd.DataFrame([[trouver_abcisse(1, tabQuantiteY, tabQuantiteX),
                                                        trouver_abcisse(20, tabQuantiteY, tabQuantiteX),
                                                        trouver_abcisse(80, tabQuantiteX, tabQuantiteY)]],
                                                      columns=["1% des articles", "20% des articles",
                                                               "80% des quantités"])

                    st.table(TableauIndicateurs.transpose().style.format('{:.2f}%').set_table_styles([{'selector': 'th',
                                                                                                       "props": [
                                                                                                           ("color",
                                                                                                            "black"),
                                                                                                           (
                                                                                                           "font-weight",
                                                                                                           "bold"),
                                                                                                           ("font-size",
                                                                                                            "20px")
                                                                                                       ]},
                                                           {"selector" : "td",
                                                            "props": [
                                                                ("font-size", "18px")
                                                            ]
                                                            }]).hide_columns())
                    save_path = "QuantitePerArticle " + str(3) + " Mois.xlsx"
                    dfextractQte_xlsx = to_excel(dfExtractQte3)
                    st.download_button(label='📥 Télécharger ce résultat',
                                       data=dfextractQte_xlsx,
                                       file_name=save_path)

                except Exception:
                    st.error("Analyse non disponible !")

                # endregion
            with qteperiode :

                ArticlePerQuantite = round(
                    (dfParticulieres.groupby(
                        by=["Code_article","Libellé_article"]).Quantité.sum() * 100 / dfParticulieres.Quantité.sum()),
                    2).sort_values(ascending=False)

                listeQuantite = []
                SommeQuantite = 0

                for item in ArticlePerQuantite.items():
                    SommeQuantite += item[1]
                    listeQuantite.append(SommeQuantite)
                tabQuantiteX = []
                for i in range(1, len(listeQuantite) + 1):
                    tabQuantiteX.append(round(((i / len(listeQuantite)) * 100), 2))

                CodeArticles = []
                LibelleArticles = []
                for i in RepartitionArticle.index:
                    CodeArticles.append(i[0])
                    LibelleArticles.append(i[1])

                dfExtractQtePerperiode = pd.DataFrame(list(zip(CodeArticles, LibelleArticles, tabQuantiteX, listeQuantite)),
                                         columns=["Code_article", "Libellé_article", "Pourcentage d'article",
                                                  "Pourcentage de mouvement"])

                dfExtractQtePerperiode["Pourcentage de mouvement"] = dfExtractQtePerperiode["Pourcentage de mouvement"].astype(str)
                dfExtractQtePerperiode["Pourcentage d'article"] = dfExtractQtePerperiode["Pourcentage d'article"].astype(str)
                dfExtractQtePerperiode["Pourcentage de mouvement"] = dfExtractQtePerperiode["Pourcentage de mouvement"] + " %"
                dfExtractQtePerperiode["Pourcentage d'article"] = dfExtractQtePerperiode["Pourcentage d'article"] + " %"

                listeQuantite.append(0)
                tabQuantiteX.append(0)
                listeQuantite.sort()
                tabQuantiteX.sort()
                tabQuantiteY = np.array(listeQuantite)

                abcisse80Quantite = trouver_abcisse(80, tabQuantiteX, tabQuantiteY)
                tab20QuantiteX = np.array([20, 20, 20])
                tab20QuantiteY = np.array([0, 40, 100])

                tab20Quantitexx = np.array([abcisse80Quantite, abcisse80Quantite, abcisse80Quantite])
                tab20Quantiteyy = np.array([0, 40, 100])

                figureQuantitePerPeriode = plt.figure(figsize=(7, 3))
                plt.plot(tabQuantiteX, tabQuantiteY, color="black", linewidth=3.0)

                plt.grid(True)
                plt.xticks(np.arange(0, 110, step=10))
                plt.yticks(np.arange(0, 110, step=10))
                plt.plot(tab20QuantiteX, tab20QuantiteY, color="red", linewidth=2.0, linestyle='--')
                plt.plot(tab20Quantitexx, tab20Quantiteyy, color="blue", linewidth=2.0, linestyle='--')
                plt.ylim((0, 110))
                plt.xlim((0, 110))

                plt.plot()
                plt.xlabel("Articles ( % )")
                plt.ylabel("Quantités ( % )")

                st.pyplot(figureQuantitePerPeriode)

                # region Tableau Indicateurs

                TableauIndicateurs = pd.DataFrame([[trouver_abcisse(1, tabQuantiteY, tabQuantiteX),
                                                    trouver_abcisse(20, tabQuantiteY, tabQuantiteX),
                                                    trouver_abcisse(80, tabQuantiteX, tabQuantiteY)]],
                                                  columns=["1% des articles", "20% des articles",
                                                           "80% des quantités"])

                st.table(TableauIndicateurs.transpose().style.format('{:.2f}%').set_table_styles([{'selector': 'th',
                                                                                                   "props": [
                                                                                                       ("color",
                                                                                                        "black"),
                                                                                                       (
                                                                                                       "font-weight",
                                                                                                       "bold"),
                                                                                                       ("font-size",
                                                                                                        "20px")
                                                                                                   ]},
                                                           {"selector" : "td",
                                                            "props": [
                                                                ("font-size", "18px")
                                                            ]
                                                            }]).hide_columns())

                save_path = "QuantitePerArticle " + str(optionDate) + " Mois.xlsx"
                dfQtePerPeriode_xlsx = to_excel(dfExtractQtePerperiode)
                st.download_button(label='📥 Télécharger ce résultat',
                                   data=dfQtePerPeriode_xlsx,
                                   file_name=save_path)

                    # endregion
            # endregion


        #endregion

