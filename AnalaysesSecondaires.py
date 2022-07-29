import numpy as np
import datetime
import pandas as pd
import plotly.express as px
from datetime import timedelta
from io import BytesIO
import streamlit as st

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





class AnalysesSecondaires():
    def __init__(self):
        return

    def __call__(self, Mouvements, stocks):

        if Mouvements is not None :
            # region Erreur d'inventaire
            # if Mouvements is not None :
            #
            #
            #
            #     #if Modif. GEI par l'inventaire existe dans les libellés de mouvements :
            #     #sinon traiter tous les libbeles des mouvements et trouver les tops 6 les plus impactants
            #     st.header("Erreur de l'inventaire")
            #     data = Mouvements.copy()
            #     data[np.logical_or((data["Libellé_mouvement"] == "Modif. GEI par l'inventaire"), (data["Libellé_mouvement"] == "Modification quantité/poids"))]["Code_mouvement"] = 410
            #     data[data["Libellé_mouvement"] == "Expédition destinataire"]["Code_mouvement"] = 500
            #     data[data["Libellé_mouvement"] == "Réception fournisseur"]["Code_mouvement"] = 100
            #     CodeMouvementModif1 = 0
            #     CodeMouvementModif2 = 0
            #     if "Modif. GEI par l'inventaire" in data["Libellé_mouvement"].unique().tolist():
            #         CodeMouvementModif1 = data[data["Libellé_mouvement"] == "Modif. GEI par l'inventaire"]["Code_mouvement"].tolist()[0]
            #     if "Modification quantité/poids" in data["Libellé_mouvement"].unique().tolist():
            #         CodeMouvementModif2 = data[data["Libellé_mouvement"] == "Modification quantité/poids"]["Code_mouvement"].tolist()[0]
            #     cols = data.columns
            #
            #     inv = data[np.logical_or((data[cols[3]] == CodeMouvementModif1), (data[cols[3]] == CodeMouvementModif2))]
            #     recep = data[data[cols[3]] == 100]
            #     exped = data[data[cols[3]] == 500]
            #
            #     def IQR(df, q):
            #         # IQR method :
            #         q_max = df[cols[6]].quantile(q, interpolation='lower')
            #         q_min = df[cols[6]].quantile(0.25)
            #         lim_min = q_min - 1.5 * (q_max - q_min)
            #         lim_max = q_max + 1.5 * (q_max - q_min)
            #         print('Intervalle : ', lim_min, lim_max)
            #         return list(df[(df[cols[6]] <= lim_min) | (df[cols[6]] >= lim_max)].index)
            #
            #     ind_to_drop = IQR(inv, 0.75) + IQR(recep, 0.9) + IQR(exped, 0.9)
            #     print('Proportion of operations being removed : ', len(ind_to_drop) / (len(inv) + len(recep) + len(exped)))
            #
            #     data.drop(ind_to_drop, axis='index', inplace=True)
            #
            #     #Création d'une colonne pour les erreurs d'inventaire
            #     refs = inv.groupby(cols[1]).sum()[[cols[6]]].rename(columns={cols[6]: 'error_inv'})
            #
            #     # Volume et nombre de reception
            #     recep = data[data[cols[3]] == 100][[cols[1], cols[6]]]
            #     vol_recep = recep.groupby(cols[1]).sum()[[cols[6]]].rename(columns={cols[6]: 'volume de reception'})
            #     nb_recep = recep.groupby(cols[1]).count()[[cols[6]]].rename(columns={cols[6]: 'nombre de reception'})
            #
            #     # Volume et nombre d'éxpédition
            #     exped = data[data[cols[3]] == 500][[cols[1], cols[6]]]
            #     vol_exped = exped.groupby(cols[1]).sum()[[cols[6]]].rename(columns={cols[6]: "volume d'expédition"})
            #     nb_exped = exped.groupby(cols[1]).count()[[cols[6]]].rename(columns={cols[6]: "nombre d'expédition"})
            #
            #     #Grosses expéditions
            #     huge = exped[cols[6]].quantile(0.98)
            #     nb_huge_exped = exped[exped[cols[6]] >= huge].groupby(cols[1])[cols[6]]
            #     huge_expeds = nb_huge_exped.count()
            #     huge_expeds.rename('nombre de grosses expéditions', inplace=True)
            #
            #     #Grosses réceptions
            #     huge = recep[cols[6]].quantile(0.85)
            #     nb_huge_recep = recep[recep[cols[6]] >= huge].groupby(cols[1])[cols[6]]
            #     huge_receps = nb_huge_recep.count()
            #     huge_receps.rename('nombre de grosses réceptions', inplace=True)
            #
            #
            #     refs = pd.merge(refs, vol_recep, left_index=True, right_index=True, how='outer')
            #     refs = pd.merge(refs, nb_recep, left_index=True, right_index=True, how='outer')
            #     refs = pd.merge(refs, vol_exped, left_index=True, right_index=True, how='outer')
            #     refs = pd.merge(refs, nb_exped, left_index=True, right_index=True, how='outer')
            #
            #     refs = pd.merge(refs, huge_receps, left_index=True, right_index=True, how='outer')
            #     refs = pd.merge(refs, huge_expeds, left_index=True, right_index=True, how='outer')
            #     refs = refs.fillna(0)
            #
            #     refs = refs[refs["error_inv"] != 0]
            #     spiderVueMatrix = refs.corr().iloc[0].to_frame()
            #     spiderVueMatrix["error_inv"] = abs(spiderVueMatrix["error_inv"])
            #
            #     spiderVueMatrix = spiderVueMatrix.transpose()
            #
            #
            #
            #     spiderVueMatrix = spiderVueMatrix[['volume de reception', 'nombre de reception','nombre de grosses réceptions',
            #                                        "volume d'expédition", "nombre d'expédition", 'nombre de grosses expéditions']]
            #     spiderVueMatrix = spiderVueMatrix.fillna(0)
            #
            #     theta = spiderVueMatrix.columns
            #     fig = px.line_polar(spiderVueMatrix, spiderVueMatrix.iloc[0].tolist(), theta, line_close=True,range_r=[0,0.6])
            #     fig.update_traces(fill='toself')
            #     fig.update_layout(
            #         showlegend=False,
            #         polar=dict(
            #             angularaxis=dict(
            #                 tickfont_size=18
            #             )
            #         ))
            #     st.plotly_chart(fig, use_container_width=True)
            #
           #endregion




            #region Articles en potentielle rupture
            if stocks is not None :

                st.header("Articles en potentielle rupture")
                DureeChoisi = st.selectbox("Choisir un intervalle de mois", [3,6,9,12])


                mouvements = Mouvements.copy()
                stock = stocks.copy()
                stock = stock.rename(columns={"Quantité": "Quantité en stock"})
                mouvements = mouvements.rename(columns={"Quantité": "Quantité en mouvement"})
                mouvements =  mouvements[mouvements["Libellé_mouvement"] == "Expédition destinataire" ]



                endDate = mouvements.Date_création.max()
                startDate = setDateDebutWithMonth(DureeChoisi, endDate)
                MouvementPerDate = mouvements[mouvements["Date_création"].isin(pd.date_range(start=startDate, end=endDate))]
                MouvementPerDate["Premier_mouvement"] = ""
                MouvementPerDate["Dernier_mouvement"] = ""
                for i in MouvementPerDate.index:
                    MouvementPerDate["Premier_mouvement"][i] = datetime.datetime.strftime(MouvementPerDate[MouvementPerDate["Code_article"] ==MouvementPerDate["Code_article"][i]].Date_création.min(),"%d-%m-%Y")
                    MouvementPerDate["Dernier_mouvement"][i] = datetime.datetime.strftime(MouvementPerDate[MouvementPerDate["Code_article"] ==MouvementPerDate[ "Code_article"][i]].Date_création.max(),"%d-%m-%Y")

                MouvementPerArticle = MouvementPerDate[["Code_article", "Libellé_article", "Premier_mouvement", "Dernier_mouvement"]].value_counts().to_frame()
                MouvementPerArticle = MouvementPerArticle.rename(columns={0: "Nombre de mouvements"})


                startStockDate = stock.Date_création.max() - timedelta(days= 30 * DureeChoisi)
                endStockDate = stock.Date_création.max()
                StockPerDate = stock[stock["Date_création"].isin(pd.date_range(start=startStockDate, end=endStockDate))]
                stockMois = StockPerDate.groupby(["Code_article","Date_création"])["Quantité en stock"].sum().to_frame()
                MouvementPerDate["Quantité en mouvement"] = abs(MouvementPerDate["Quantité en mouvement"])

                mouvementMois = MouvementPerDate.groupby("Code_article")["Quantité en mouvement"].sum().to_frame()
                Mouvements_Stock_Mois = pd.merge(stockMois, mouvementMois, left_index=True, right_index=True)
                Mouvements_Stock_Mois = pd.merge(Mouvements_Stock_Mois, MouvementPerArticle, left_index=True, right_index=True)


                ArticlesAnalyses = Mouvements_Stock_Mois.sort_values(by="Quantité en stock", ascending=False)
                ArticlesAnalyses["Quantité restante"] = round(ArticlesAnalyses["Quantité en stock"] - abs(ArticlesAnalyses["Quantité en mouvement"]),2)

                CodesArticles = []
                Dates_Stock = []
                LibellesArticles = []
                PremiersMouvements = []
                DerniersMouvements = []

                for i in ArticlesAnalyses.index.tolist():
                    CodesArticles.append(i[0])
                    Dates_Stock.append(i[1])
                    LibellesArticles.append(i[2])
                    PremiersMouvements.append(i[3])
                    DerniersMouvements.append(i[4])

                ArticlesAnalyses["Code_article"] = CodesArticles
                ArticlesAnalyses["Libellé_article"] = LibellesArticles
                ArticlesAnalyses["Premier mouvement"] = PremiersMouvements
                ArticlesAnalyses["Dernier mouvement"] = DerniersMouvements
                ArticlesAnalyses["Date_création"] = Dates_Stock

                DfResultats = pd.DataFrame(list(zip(ArticlesAnalyses["Code_article"].tolist(),
                                                    ArticlesAnalyses["Libellé_article"].tolist(),

                                                    ArticlesAnalyses["Premier mouvement"].tolist(),
                                                    ArticlesAnalyses["Dernier mouvement"].tolist(),
                                                    ArticlesAnalyses["Quantité en stock"].tolist(),
                                                    ArticlesAnalyses["Nombre de mouvements"].tolist(),
                                                    ArticlesAnalyses["Quantité en mouvement"].tolist(),
                                                    ArticlesAnalyses["Date_création"].tolist()

                                                    )),
                                           columns=["Code Article",
                                                    "Libellé Article",

                                                    "Premier mouvement",
                                                    "Dernier mouvement",
                                                    "Quantité en stock",
                                                    "Nombre de mouvements",
                                                    "Quantités gérées par les mouvements",
                                                    "Date de stock"])

                DfResultats["Quantité potentielle restante"] = 0
                DfResultats["Durée potentielle restante"] = 0
                for i in DfResultats.index:
                    code = DfResultats["Code Article"][i]
                    if DfResultats["Date de stock"][i] < mouvements["Date_création"].max():
                        mvtInterm = mouvements[np.logical_and((mouvements["Code_article"] == code), (mouvements["Date_création"].isin(pd.date_range(
                                                                        start=DfResultats["Date de stock"][i],
                                                                        end=endDate))))]
                        somme = DfResultats["Quantité en stock"][i] + mvtInterm["Quantité en mouvement"].sum()
                        DfResultats["Quantité potentielle restante"][i] = somme
                    else:
                        DfResultats["Quantité potentielle restante"][i] = stock[stock["Code_article"] == code]["Quantité en stock"].sum()

                for i in DfResultats.index :
                    DfResultats["Durée potentielle restante"][i] = round(DureeChoisi * 30 * DfResultats["Quantité potentielle restante"][i] / abs(DfResultats["Quantités gérées par les mouvements"][i]))





                DfResultats = DfResultats[["Code Article","Libellé Article","Quantité potentielle restante","Durée potentielle restante","Premier mouvement","Dernier mouvement",
                                           "Quantités gérées par les mouvements","Nombre de mouvements"]]


                DfResultats["Quantités gérées par les mouvements"] = abs(DfResultats["Quantités gérées par les mouvements"])
                DfResultats["Durée potentielle restante"] = abs(DfResultats["Durée potentielle restante"])
                DfResultats["Quantité potentielle restante"] = abs(DfResultats["Quantité potentielle restante"])

                DfResultats = DfResultats[DfResultats["Quantité potentielle restante"] != 0]

                DfResultats = DfResultats.sort_values(by="Durée potentielle restante",ascending = True)

                DfResultats["Durée potentielle restante"] = DfResultats["Durée potentielle restante"].astype(str)
                DfResultats["Durée potentielle restante"] = DfResultats["Durée potentielle restante"] + " j"
                DfResultats = DfResultats.reset_index(drop=True)

                st.table(DfResultats.head(15).style.set_table_styles([{'selector': 'th',
                                        "props": [
                                            ("color", "black"),
                                            ("font-weight", "bold"),
                                            ("font-size", "18px")
                                        ]},
                                                               {"selector" : "td",
                                                                "props": [
                                                                    ("font-size", "18px")
                                                                ]
                                                                },
                                                                       {"selector": "tbody>tr>:nth-child(4)",
                                                                        "props": [("font-weight", "bold")]},
                                                                       {"selector": "tbody>tr>:nth-child(5)",
                                                                        "props": [("font-weight", "bold")]}]))
                df_xlsx = to_excel(DfResultats)
                st.download_button(label='📥 Télécharger ce résultat',
                                   data=df_xlsx,
                                   file_name='ArticleEnPotentielleRupture.xlsx')







            if Mouvements is not None and stocks is None :
                st.header("Articles en potentielle rupture")
                OptionDate = st.selectbox("Choisir un intervalle de mois", [3, 6, 9, 12])
                dataframe = Mouvements.copy()
                startDate = dataframe.Date_création.max() - timedelta(days=30 * OptionDate)
                endDate = dataframe.Date_création.max()
                MouvementPerDate = dataframe[dataframe["Date_création"].isin(pd.date_range(start=startDate, end=endDate))]
                MouvementPerDate = MouvementPerDate[MouvementPerDate["Libellé_mouvement"] == "Expédition destinataire"]
                MouvementPerDate["Quantité"] = abs(MouvementPerDate["Quantité"])
                NombreMouvementPerDate = MouvementPerDate.groupby(["Code_article"])[
                    "Code_article"].value_counts().to_frame()
                NombreMouvementPerDate = NombreMouvementPerDate.rename(columns={"Code_article": "Nombre de mouvement"})
                NombreMouvementPerDate["Code_article"] = ""

                for i in NombreMouvementPerDate.index:
                    NombreMouvementPerDate["Code_article"][i] = i[0]
                NombreMouvementPerDate = NombreMouvementPerDate.reset_index(drop=True)
                MouvementPerDate = MouvementPerDate.groupby(["Code_article", "Libellé_article"])[
                    "Quantité"].sum().to_frame()
                ListeCodes = []
                ListeLibelle = []
                for indice in MouvementPerDate.index:
                    ListeCodes.append(indice[0])
                    ListeLibelle.append(indice[1])
                MouvementPerDate["Code_article"] = ListeCodes
                MouvementPerDate["Libellé_article"] = ListeLibelle
                MouvementPerDate = MouvementPerDate[["Code_article", "Libellé_article", "Quantité"]]
                MouvementPerDate = MouvementPerDate.sort_values(by="Quantité", ascending=False)
                MouvementPerDate = MouvementPerDate.reset_index(drop=True)
                MouvementPerDate["Quantité/Mois"] = ""
                for i in MouvementPerDate.index:
                    MouvementPerDate["Quantité/Mois"][i] = str(round(MouvementPerDate["Quantité"][i] / OptionDate))
                ArticleEnPotentielleRuptureDf = MouvementPerDate.merge(NombreMouvementPerDate, on="Code_article")
                ArticleEnPotentielleRuptureDf = ArticleEnPotentielleRuptureDf[["Code_article", "Libellé_article", "Nombre de mouvement", "Quantité", "Quantité/Mois"]]
                ArticleEnPotentielleRuptureDf = ArticleEnPotentielleRuptureDf.rename(columns = {"Nombre de mouvement" : "Nb de mouvement"})
                for i in ArticleEnPotentielleRuptureDf.index :
                    ArticleEnPotentielleRuptureDf["Quantité"][i] = "{:,}".format(ArticleEnPotentielleRuptureDf["Quantité"][i]).replace(',', ' ')
                    ArticleEnPotentielleRuptureDf["Nb de mouvement"][i] = "{:,}".format(ArticleEnPotentielleRuptureDf["Nb de mouvement"][i]).replace(',', ' ')
                    ArticleEnPotentielleRuptureDf["Quantité/Mois"][i] = str(    "{:,}".format(int(ArticleEnPotentielleRuptureDf["Quantité/Mois"][i])).replace(',', ' '))


                ArticleEnPotentielleRuptureDf["Code_article"] = ArticleEnPotentielleRuptureDf["Code_article"].astype(str)
                st.table(ArticleEnPotentielleRuptureDf.head(15).style.set_table_styles([{'selector': 'th',
                                                                       "props": [
                                                                           ("color", "black"),
                                                                           ("font-weight", "bold"),
                                                                           ("font-size", "18px"),
                                                                           ("word-wrap" , "break-word"),
                                                                           ("text-align" , "center")
                                                                       ]},
                                                                      {"selector": "td",
                                                                       "props": [
                                                                           ("font-size", "18px"),
                                                                           ("word-wrap" , "break-word"),
                                                                           ("text-align" , "center")
                                                                       ]
                                                                       },
                                                                       {"selector": "tbody>tr>:nth-child(6)",
                                                                        "props": [("font-weight", "bold")]}]))

                dfRupture_xlsx = to_excel(ArticleEnPotentielleRuptureDf)
                st.download_button(label='📥 Télécharger ce résultat',
                                   data=dfRupture_xlsx,
                                   file_name='ArticleEnPotentielleRupture.xlsx')














            #endregion




            #region Stock Mort
            if Mouvements is not None and stocks is not None:


                st.header("Stock Mort")
                dataframe = Mouvements.copy()
                dataframe = dataframe[dataframe['Libellé_mouvement'] == "Expédition destinataire"]
                stock = stocks.copy()



                optionDate = st.selectbox("Choisir un intervalle de mois",[12,15,18,21,24],index=0)
                st.markdown("")

                startDate = setDateDebutWithMonth(optionDate, dataframe.Date_création.max())
                endDate = dataframe.Date_création.max()
                dataframeMois = dataframe[dataframe["Date_création"].isin(pd.date_range(start=startDate, end=endDate))]


                dataframeOld = dataframe[dataframe["Date_création"].isin(pd.date_range(start=dataframe.Date_création.min(), end=startDate))]

                ListeTousLesArticles = dataframe["Libellé_article"].unique().tolist()
                ListeCodeArticles = dataframe["Code_article"].unique().tolist()

                ListeArticlesEnRotation = dataframeMois["Libellé_article"].unique().tolist()
                ListeCodeArticleEnRotation = dataframeMois["Code_article"].unique().tolist()

                ListeArticlesLibelleCheck = dataframeOld["Libellé_article"].unique().tolist()
                ListeArticlesCodeCheck = dataframeOld["Code_article"].unique().tolist()


                StockMortLibelle = []
                StockMortCode = []

                for article in ListeTousLesArticles:
                    if article not in ListeArticlesEnRotation and article in ListeArticlesLibelleCheck:
                        StockMortLibelle.append(article)
                for article in ListeCodeArticles:
                    if article not in ListeCodeArticleEnRotation and article in ListeArticlesCodeCheck:
                        StockMortCode.append(article)

                dframe = pd.DataFrame(list(zip(StockMortCode, StockMortLibelle)), columns=["Code de l'article","Libellé de l'article"])
                dframe["Quantité en stock"] = 0


                for i in dframe.index:
                    if len(stock[stock.Code_article == dframe["Code de l'article"][i]]) > 0:
                        dfstock = stock[stock.Code_article == dframe["Code de l'article"][i]]
                        if dfstock.Date_création.max() < dataframe.Date_création.max():
                            dfInt = dataframe[np.logical_and((dataframe["Code_article"] == dframe["Code de l'article"][i]), (
                                dataframe["Date_création"].isin(pd.date_range(start=dfstock.Date_création.max(),
                                                                              end=dataframe.Date_création.max()))))]
                            dframe["Quantité en stock"][i] = dfstock["Quantité"].sum() + dfInt["Quantité"].sum()
                        else:
                            dframe["Quantité en stock"][i] = dfstock["Quantité"].sum()





                dframe = dframe.sort_values(by="Quantité en stock", ascending=False)
                dframe = dframe.rename(columns={"Quantité en stock": "Quantité potentielle en stock"})
                dframe = dframe[dframe["Quantité potentielle en stock"] != 0]
                dframe = dframe.reset_index(drop=True)

                st.markdown("""
                                    <style>
                                        .markdown-font { font-size : 20px }
                                    </style>
                                """, unsafe_allow_html=True)

                InfoStockMort = """<p class = 'markdown-font' > Le nombre d'article sans mouvement sur cette période est <b>""" + str(len(dframe)) + " articles </b> </p>"
                st.markdown(InfoStockMort, unsafe_allow_html=True)


                st.table(dframe.head(20).style.set_table_styles([{'selector': 'th',
                                        "props": [
                                            ("color", "black"),
                                            ("font-weight", "bold"),
                                            ("font-size", "18px")
                                        ]},
                                                               {"selector" : "td",
                                                                "props": [
                                                                    ("font-size", "18px")
                                                                ]
                                                                }]))
                dframe_xlsx = to_excel(dframe)
                FilePath = "StockMort" + str(optionDate) + ".xlsx"
                st.download_button(label='📥 Télécharger ce résultat',
                                   data=dframe_xlsx,
                                   file_name=FilePath)

            #endregion
