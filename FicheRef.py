import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np




#region Fonctions


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
    if (dateFin.month == 1) and (dateFin.day != months[1]):
        date = str(dateFin.year - 1) + "-12-31"
        dateFin = pd.Timestamp(date)
    if ((dateFin.month == 2) and (dateFin.year % 4 == 0) and (dateFin.day != 29)):
        date = str(dateFin.year) + "-01-31"
        dateFin = pd.Timestamp(date)

    if ((dateFin.month == 2) and (dateFin.year % 4 != 0) and (dateFin.day != 28)):
        date = str(dateFin.year) + "-01-31"
        dateFin = pd.Timestamp(date)

    if (dateFin.month != 2) and (dateFin.day != months[dateFin.month]):
        date = str(dateFin.year) + "-" + str(dateFin.month - 1) + "-" + str(months[dateFin.month - 1])
        dateFin = pd.Timestamp(date)
    return dateFin
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

def HistNbreMvtPerDate(ReferenceDf) :
    dfs = []
    types_mvt = ReferenceDf.Libellé_mouvement.value_counts().index
    for type_mvt in types_mvt:
        df = ReferenceDf[ReferenceDf['Libellé_mouvement'] == type_mvt]
        df = df.groupby(df.index).count()
        dfs.append(df.Quantité)
    flux_acti = pd.concat(dfs, axis=1)
    flux_acti.columns = types_mvt
    flux_acti.fillna(0, inplace=True)
    flux_acti = flux_acti.astype(int)
    flux_acti.index = pd.to_datetime(flux_acti.index)

    return flux_acti

def HistQtePerDate(ReferenceDf) :
    dfs = []
    types_mvt = ReferenceDf.Libellé_mouvement.value_counts().index
    for type_mvt in types_mvt:
        df = ReferenceDf[ReferenceDf['Libellé_mouvement'] == type_mvt]
        df = df.groupby(df.index).sum()
        dfs.append(df.Quantité)
    flux_acti = pd.concat(dfs, axis=1)
    flux_acti.columns = types_mvt
    flux_acti.fillna(0, inplace=True)
    flux_acti = flux_acti.astype(int)
    flux_acti.index = pd.to_datetime(flux_acti.index)

    return flux_acti



    return

#endregion

months = {
    1:31,
    2:28,
    3:31,
    4:30,
    5:31,
    6:30,
    7:31,
    8:31,
    9:30,
    10:31,
    11:30,
    12:31
}





class FicheRef():
    def __init__(self):
        return

    def __call__(self, Mouvements,stock):
        if Mouvements is not None :
            reference = st.text_input("Entrer le code d'article à cibler ...")
            Mouvements["Code_article"] = Mouvements["Code_article"].astype(str)

            if reference != "":
                if reference in Mouvements["Code_article"].tolist():
                    dataframe = Mouvements.copy()

                    #region Preparation de la base
                    PremierMouvement =  dataframe[dataframe["Code_article"] == reference]["Date_création"].min()
                    DernierMouvement =  dataframe[dataframe["Code_article"] == reference]["Date_création"].max()
                    NombreMouvement = len(dataframe[dataframe["Code_article"] == reference])
                    libelle = dataframe[dataframe["Code_article"] == reference]["Libellé_article"].tolist()[0]
                    dataframe["Quantité"] = abs(dataframe["Quantité"])
                    Quantite = dataframe[dataframe["Code_article"] == reference]["Quantité"].sum()


                    TailleBaseAEtudier = (DernierMouvement - PremierMouvement).days
                    if TailleBaseAEtudier < 365 :
                        ReferenceDf = dataframe[dataframe["Code_article"] == reference]
                    else:
                        ReferenceDf = dataframe[np.logical_and((dataframe["Code_article"] == reference),
                                                               (dataframe["Date_création"]
                                                                   .isin(pd.date_range(
                                                                   start=setDateDebutWithMonth(12,dataframe.Date_création.max()),
                                                                   end=dataframe.Date_création.max()))))]


                    ReferenceDf.index = pd.to_datetime(ReferenceDf["Date_création"],format="%d%m%Y")
                    #endregion



                    #region Informations générales sur la référence
                    st.header(body="Informations générales sur la référence " + str(reference) )





                    PremierMouvement = datetime.datetime.strftime(PremierMouvement, "%d-%m-%Y")
                    DernierMouvement = datetime.datetime.strftime(DernierMouvement, "%d-%m-%Y")

                    #La dataframe des résultats
                    ListeInformations = [libelle, NombreMouvement, Quantite, PremierMouvement, DernierMouvement]
                    ListeColonne = ["Libellé de l'article", "Le nombre total des mouvements",
                                    "Quantités gérées par les mouvements", "La date du premier mouvement",
                                    "La date du dernier mouvement"]

                    InformationsRefDf = pd.DataFrame([ListeInformations], columns=ListeColonne)
                    InformationsRefDf["Le nombre total des mouvements"] = InformationsRefDf["Le nombre total des mouvements"].astype(object)
                    for i in InformationsRefDf.index:
                        InformationsRefDf["Le nombre total des mouvements"][i] = "{:,}".format( InformationsRefDf["Le nombre total des mouvements"][i]).replace(',', ' ')
                        InformationsRefDf["Quantités gérées par les mouvements"][i] = "{:,}".format( InformationsRefDf["Quantités gérées par les mouvements"][i]).replace(',', ' ')




                    st.table(InformationsRefDf.style.set_table_styles([{'selector': 'th',
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



                    #region  Details sur les mouvements de la reference
                    st.header(body = "Details sur les mouvements de la référence " + str(reference))

                    Mvtdf = ReferenceDf["Libellé_mouvement"].value_counts().to_frame().sort_values(by=["Libellé_mouvement"])
                    QtDf = ReferenceDf.groupby("Libellé_mouvement")["Quantité"].sum().to_frame().sort_values(
                        by=["Libellé_mouvement"])
                    Mvtdf = Mvtdf.rename(columns={"Libellé_mouvement": "Nombre de mouvement"})
                    results = pd.merge(Mvtdf, QtDf, on=Mvtdf.index)
                    results = results.rename(columns={"key_0": "Libellé_mouvement"})
                    resultats = results[["Libellé_mouvement", "Nombre de mouvement", "Quantité"]]



                    resultats["Quantité"] = resultats["Quantité"].astype(object)
                    resultats["Nombre de mouvement"] = resultats["Nombre de mouvement"].astype(object)

                    for i in resultats.index :
                        resultats["Quantité"][i] = "{:,}".format( resultats["Quantité"][i]).replace(',', ' ')
                        resultats["Nombre de mouvement"][i] = "{:,}".format( resultats["Nombre de mouvement"][i]).replace(',', ' ')


                    # region hide index
                    # CSS to inject contained in a string
                    hide_table_row_index = """
                                                                  <style>
                                                                  tbody th {display:none}
                                                                  .blank {display:none}
                                                                  </style>
                                                                  """

                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)

                    # endregion
                    st.table(resultats.style.set_table_styles([{'selector': 'th',
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



                    #region Histogramme Nombre de mouvement par date
                    st.header("Analyse par nombre de mouvement")
                    hist_nb_mvt = HistNbreMvtPerDate(ReferenceDf)

                    st.text("En nombre de mouvement")
                    st.bar_chart(hist_nb_mvt.groupby(pd.Grouper(freq="M")).sum())
                    #endregion



                    #region Histogramme Quantité par date
                    with st.expander("ANALYSE PAR VOLUME DE QUANTITES GEREES PAR LES MOUVEMENTS") :

                        histQteDf = Mouvements.copy()[Mouvements.copy()["Code_article"] == reference]
                        histQteDf.index = pd.to_datetime(histQteDf["Date_création"],format="%d%m%Y")

                        PremierMouvement = histQteDf[histQteDf["Code_article"] == reference]["Date_création"].min()
                        DernierMouvement = histQteDf[histQteDf["Code_article"] == reference]["Date_création"].max()

                        TailleBaseAEtudier = (DernierMouvement - PremierMouvement).days
                        if TailleBaseAEtudier < 365:
                            histQteDf = histQteDf[histQteDf["Code_article"] == reference]
                        else:
                            histQteDf = histQteDf[np.logical_and((histQteDf["Code_article"] == reference),
                                                                   (histQteDf["Date_création"]
                                                                       .isin(pd.date_range(
                                                                       start=setDateDebutWithMonth(12,histQteDf.Date_création.max()),
                                                                       end=histQteDf.Date_création.max()))))]





                        hist_qte_mvt = HistQtePerDate(histQteDf)
                        st.text("En nombre de quantité")
                        st.bar_chart(hist_qte_mvt.groupby(pd.Grouper(freq="M")).sum())
                    #endregion


                    if  stock is not None :
                        #region Courbe d'évolution des mouvements de cette reference

                        st.header("Evolution du stock de la référénce " + reference)
                        stock["Code_article"] = stock["Code_article"].astype(str)
                        stocks = stock.copy()
                        if (stocks.Code_article.dtypes == 'float64'):
                            stocks.Code_article = stocks.Code_article.astype(str)
                            for i in stocks.index:
                                stocks["Code_article"][i] = str(round(float(stocks["Code_article"][i])))

                        stocks["Code_article"] = stocks["Code_article"].astype(str)

                        ListeArticlesStock = stocks.Code_article.unique().tolist()
                        ListeQteStock = []
                        ListeDate = []
                        for article in ListeArticlesStock:
                            ListeQteStock.append(stocks[stocks["Code_article"] == article]["Quantité"].sum())
                            ListeDate.append(stocks[stocks["Code_article"] == article]["Date_création"].tolist()[0])
                        stockDf = pd.DataFrame(list(zip(ListeArticlesStock, ListeQteStock, ListeDate)),
                                               columns=["Code_article", "Quantité", "Date_création"])
                        MouvementsDf = Mouvements[["Code_article", "Quantité", "Date_création"]]
                        endDate = MouvementsDf.Date_création.max()
                        startDate = MouvementsDf.Date_création.min()

                        startDate = setDateDebut(startDate)
                        endDate = setDateFin(endDate)

                        MouvementsDf = MouvementsDf[MouvementsDf["Date_création"].isin(
                            pd.date_range(start=startDate,
                                          end=endDate))]
                        if reference not in stockDf["Code_article"].tolist():
                            st.error("La référence n'existe pas dans le fichier stock fourni !! ")
                        else :
                            Date_Stock = stockDf[stockDf["Code_article"] == reference]["Date_création"].tolist()[0]

                            Df = MouvementsDf[MouvementsDf["Code_article"] == reference].append(
                                stockDf[stockDf["Code_article"] == reference], ignore_index=True)
                            qte = stockDf[stockDf["Code_article"] == reference]["Quantité"]
                            q = qte
                            Df = Df.sort_values(by="Date_création", ascending=False)
                            Df["Qte"] = 0

                            if (Df.Date_création.max() == Date_Stock):

                                for i in Df.index:
                                    if (Df["Date_création"][i] != Date_Stock):
                                        Df["Qte"][i] = qte - Df['Quantité'][i]
                                        qte = Df["Qte"][i]
                                DfFinale = Df.copy()
                            else:
                                dfAvantStock = Df[
                                    Df["Date_création"].isin(pd.date_range(start=Df["Date_création"].min(), end=Date_Stock))]
                                dfAvantStock = dfAvantStock.sort_values(by="Date_création", ascending=False)
                                qteEnstock = qte
                                for j in dfAvantStock.index:
                                    if dfAvantStock["Date_création"][j] == Date_Stock:
                                        dfAvantStock["Qte"][j] = qteEnstock
                                    if dfAvantStock["Date_création"][j] != Date_Stock:
                                        dfAvantStock["Qte"][j] = qte - Df["Quantité"][j]
                                        qte = dfAvantStock["Qte"][j]

                                dfApresStock = Df[
                                    Df["Date_création"].isin(pd.date_range(start=Date_Stock, end=Df["Date_création"].max()))]
                                dfApresStock = dfApresStock.sort_values(by="Date_création", ascending=True)
                                for k in dfApresStock.index:
                                    if dfApresStock["Date_création"][k] != Date_Stock:
                                        dfApresStock["Qte"][k] = qteEnstock + Df["Quantité"][k]
                                        qteEnstock = dfApresStock["Qte"][k]
                                dfApresStock = dfApresStock[dfApresStock["Date_création"] != Date_Stock]
                                DfFinale = dfAvantStock.append(dfApresStock, ignore_index=True)

                            for i in DfFinale.index :
                                if DfFinale["Date_création"][i] == Date_Stock :
                                    DfFinale["Qte"][i] = q


                            DfFinale = DfFinale.sort_values(by="Date_création", ascending=False)
                            ListIndice = DfFinale.index.tolist()
                            ListIndice.reverse()
                            for i in range(1, len(ListIndice)):
                                DfFinale["Date_création"][ListIndice[i]] = DfFinale["Date_création"][
                                                                               ListIndice[i]] + timedelta(seconds=i)

                            DfFinale.index = DfFinale["Date_création"]
                            st.line_chart(DfFinale["Qte"])

                            with st.expander("Détails sur l'évolution du stock de la référence " + reference):
                                DfFinale = DfFinale[["Code_article", "Date_création", "Qte"]]
                                st.table(DfFinale.style.set_table_styles([{'selector': 'th',
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


                        #endregion



                    #region Catégorisation d'une référence

                    st.header("Catégorisation de la référence " + reference)
                    optionDate = st.selectbox("Choisir un intervalle de mois", [1,3,6,9,12],index=1)

                    df = Mouvements.copy()

                    endDate = setDateFin(df.Date_création.max())
                    startDate = setDateDebutWithMonth(optionDate, df.Date_création.max())





                    df["Quantité"] = abs(df["Quantité"])
                    dataframeMois = df[df["Date_création"].isin(pd.date_range(start=startDate, end=endDate))]
                    dataframeMois = dataframeMois[dataframeMois["Libellé_mouvement"] == "Expédition destinataire"]
                    ArticlePerQuantite = dataframeMois.groupby(["Code_article"])["Quantité"].apply(lambda x: x.sum())
                    ArticlePerMouvement = dataframeMois["Code_article"].value_counts()
                    Df = pd.merge(ArticlePerQuantite, ArticlePerMouvement, right_index=True, left_index=True)

                    Df.columns = ["Quantite", "Nombre de mouvement"]
                    Df["Code_article"] = Df.index

                    xlim = round(Df["Nombre de mouvement"].mean()) * 2
                    ylim = round(Df["Quantite"].mean())

                    st.markdown(
                        f'<p style="color: black; font-size: 15px ; font-weight : bold; ">Dans mon magasin, je considère un article comme très productif s il est caractérisé par :</p>',
                        unsafe_allow_html=True)

                    with st.form(key="my-form"):
                        col1, col2 = st.columns(2)
                        with col1:
                            nombreMvt = st.number_input(label="Un nombre de mouvements supérieur ou égal à ",
                                                        value=round(xlim / 6))
                        with col2:
                            Qte = st.number_input(label="Une quantité gérée par les mouvements supérieure ou égale à ",
                                                  value=round(ylim / 6))

                        # Centrer le bouton :p
                        cola, colb, colc, cold, cole = st.columns(5)
                        with colc:
                            submit = st.form_submit_button("Je valide mon choix")
                    xlim = nombreMvt * 6
                    ylim = Qte * 6
                    if submit:
                        if (nombreMvt * 2) != xlim:
                            xlim = nombreMvt * 2 * 3
                        if (Qte * 2) != ylim:
                            ylim = Qte * 2 * 3


                    for i in Df.index :
                        if Df["Quantite"][i] < 0 :
                            Df["Quantite"][i] = -1 * Df["Quantite"][i]


                    DataframeA = Df[np.logical_and((Df["Quantite"] >= ylim / 2), (Df["Nombre de mouvement"] >= xlim / 2))]
                    DataframeA["Catégorie"] = "A"
                    DataframeB = Df[np.logical_and((Df["Quantite"] >= ylim / 2), (Df["Nombre de mouvement"] < xlim / 2))]
                    DataframeB["Catégorie"] = "B"
                    DataframeC = Df[np.logical_and((Df["Quantite"] < ylim / 2), (Df["Nombre de mouvement"] >= xlim / 2))]
                    DataframeC["Catégorie"] = "C"
                    DataframeD = Df[np.logical_and((Df["Quantite"] < ylim / 2), (Df["Nombre de mouvement"] < xlim / 2))]
                    DataframeD["Catégorie"] = "D"

                    Mvts = Mouvements.copy()
                    Mvts = Mvts[Mvts["Libellé_mouvement"] == "Expédition destinataire"]
                    xlimite = nombreMvt
                    ylimite = Qte

                    ListeDates = []

                    datesFin = endDate
                    for i in range(optionDate):
                        fins = datesFin
                        starts = setDateDebutWithMonth(1, fins)
                        datesFin = setDateFin(starts)
                        ListeDates.append((fins, starts))
                    ListeDates.reverse()

                    dfs = []
                    for periode in ListeDates:
                        endDate = periode[0]
                        startDate = periode[1]

                        Mvts["Quantité"] = abs(Mvts["Quantité"])
                        dataframeMois = Mvts[Mvts["Date_création"].isin(pd.date_range(start=startDate, end=endDate))]
                        ArticlePerQuantite = dataframeMois.groupby(["Code_article"])["Quantité"].apply(lambda x: x.sum())
                        ArticlePerMouvement = dataframeMois["Code_article"].value_counts()
                        Df = pd.merge(ArticlePerQuantite, ArticlePerMouvement, right_index=True, left_index=True)
                        Df.columns = ["Quantite", "Nombre de mouvement"]
                        Df["Code_article"] = Df.index
                        Df["Quantite"] = abs(Df["Quantite"])
                        df = Df.copy()
                        df["Code_article"] = df.index

                        DfA = df[np.logical_and((df["Quantite"] >= ylimite), (df["Nombre de mouvement"] >= xlimite))]

                        DfB = df[np.logical_and((df["Quantite"] >= ylimite), (df["Nombre de mouvement"] < xlimite))]

                        DfC = df[np.logical_and((df["Quantite"] < ylimite), (df["Nombre de mouvement"] >= xlimite))]

                        DfD = df[np.logical_and((df["Quantite"] < ylimite), (df["Nombre de mouvement"] < xlimite))]
                        column_name = str(endDate.month) + "/" + str(startDate.year)
                        DfA[column_name] = "A"
                        DfB[column_name] = "B"
                        DfC[column_name] = "C"
                        DfD[column_name] = "D"
                        DfA = DfA[["Code_article", column_name]]
                        DfB = DfB[["Code_article", column_name]]
                        DfC = DfC[["Code_article", column_name]]
                        DfD = DfD[["Code_article", column_name]]

                        DfExported = pd.DataFrame()
                        DfExported = DfExported.append(DfA)
                        DfExported = DfExported.append(DfB)
                        DfExported = DfExported.append(DfC)
                        DfExported = DfExported.append(DfD)
                        dfs.append(DfExported)

                    DataframeExported = pd.DataFrame()
                    DataframeExported = DataframeExported.append(DataframeA)
                    DataframeExported = DataframeExported.append(DataframeB)
                    DataframeExported = DataframeExported.append(DataframeC)
                    DataframeExported = DataframeExported.append(DataframeD)
                    DataframeExported = DataframeExported[["Code_article", "Catégorie"]]
                    DataframeExported = DataframeExported.rename(columns={"Catégorie": "Catégorie sur la période donnée"})

                    DataframeExported["Code_article"] = DataframeExported["Code_article"].astype(str)

                    if reference not in DataframeExported["Code_article"].tolist():
                        DataframeExported = DataframeExported.append({"Code_article": reference}, ignore_index=True)
                        DataframeExported.fillna("D", inplace=True)
                    DataframeExported = DataframeExported[DataframeExported["Code_article"] == reference]

                    Categorie = DataframeExported['Catégorie sur la période donnée'].tolist()[0]

                    categorisationDf = pd.concat(dfs, axis=1)
                    categorisationDf.fillna("D", inplace=True)
                    categorisationDf.drop("Code_article", axis=1, inplace=True)
                    categorisationDf["Code_article"] = categorisationDf.index
                    categorisationDf.drop_duplicates()
                    categorisationDf["Code_article"] = categorisationDf["Code_article"].astype(str)

                    if reference not in categorisationDf["Code_article"].tolist():
                        categorisationDf = categorisationDf.append({"Code_article": reference}, ignore_index=True)
                        categorisationDf.fillna("D", inplace=True)

                    categorisationDf = categorisationDf[categorisationDf["Code_article"] == reference]
                    categorisationDf.drop("Code_article", axis=1, inplace=True)

                    AxeAbcisse = categorisationDf.columns.tolist()
                    AxeOrdonnee = categorisationDf.head(1).iloc[0].tolist()

                    RefCategorie = pd.DataFrame(list(zip(AxeAbcisse, AxeOrdonnee)), columns=["Période","Catégorie"])

                    st.markdown("""
                                                    <style>
                                                        .markdown-font { font-size : 20px }
                                                    </style>
                                                """, unsafe_allow_html=True)


                    CategorieRef = """<p class = 'markdown-font' > La catégorie de la référence <b>""" + reference + "</b> sur la période ciblée est <b>" + Categorie + "</b> <p>"
                    st.markdown(CategorieRef,unsafe_allow_html=True)




                    st.table(RefCategorie.style.set_table_styles([{'selector': 'th',
                                                              "props": [
                                                                  ("color", "black"),
                                                                  ("font-weight", "bold"),
                                                                  ("font-size", "20px")
                                                              ]},
                                                             {"selector": "td",
                                                              "props": [
                                                                  ("font-size", "18px")
                                                              ]
                                                              }]))



















                    #endregion



                else :
                    st.error("Cette référence n'existe pas dans la base fournie, Veuillez la vérifier !")