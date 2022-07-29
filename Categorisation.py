import streamlit as st
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

#region Fonctions

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
class Categorisation :
    def __init__(self):
        return

    #Liste déroulante : toutes les mouvements
    def __call__(self, Mouvements):
        if Mouvements is not None :
            with st.expander("EXPLICATIONS"):
                ListeCategorie = ["A","B","C","D"]
                ListeExplications = ["Forte Quantité + Forte Rotation",
                                     "Forte Quantité + Faible Rotation",
                                     "Faible Quantité + Forte Rotation",
                                     "Faible Quantité + Faible Rotation"
                                     ]
                ExplicationDf = pd.DataFrame(list(zip(ListeCategorie, ListeExplications)), columns=["Catégorie","Désignation"])
                # CSS to inject contained in a string
                hide_table_row_index = """
                                        <style>
                                        tbody th {display:none}
                                        .blank {display:none}
                                        </style>
                                        """

                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                st.table(ExplicationDf)

            startDate = setDateDebutWithMonth(3, Mouvements.Date_création.max())
            endDate = Mouvements.Date_création.max()
            startMonth = startDate.month
            endMonth = endDate.month
            startYear = startDate.year
            endYear = endDate.year
            Mouvements["Quantité"] = abs(Mouvements["Quantité"])
            dataframeMois = Mouvements[Mouvements["Date_création"].isin(pd.date_range(start=startDate, end=endDate))]
            dataframeMois = dataframeMois[dataframeMois["Libellé_mouvement"] == "Expédition destinataire"]
            ArticlePerQuantite = dataframeMois.groupby(["Code_article", "Libellé_article"])["Quantité"].apply( lambda x: x.sum())
            ArticlePerMouvement = dataframeMois[["Code_article", "Libellé_article"]].value_counts()
            Df = ArticlePerQuantite.to_frame().join(ArticlePerMouvement.to_frame())


            Df.columns = ["Quantite", "Nombre de mouvement"]
            Df["Facteur"] = 0



            xlim = round(Df["Nombre de mouvement"].mean()) * 2
            ylim = round(Df["Quantite"].mean())





            # region Magic Quadrant 3  ( Par defaut = 3 mois )
            st.markdown(f'<p style="color: black; font-size: 15px ; font-weight : bold; ">Dans mon magasin, je considère un article comme très productif s il est caractérisé par :</p>',unsafe_allow_html=True)

            with st.form(key="my-form") :
                col1, col2 = st.columns(2)
                with col1:
                    nombreMvt = st.number_input(label="Un nombre de mouvements supérieur ou égal à ", value=round(xlim/6))
                with col2:
                    Qte = st.number_input(label="Une quantité gérée par les mouvements supérieure ou égale à ",value = round(ylim/6))

                #Centrer le bouton :p
                cola, colb, colc, cold, cole = st.columns(5)
                with colc :

                    submit = st.form_submit_button("Je valide mon choix")
            xlim = nombreMvt * 6
            ylim = Qte * 6
            if submit :
                if (nombreMvt *2) != xlim:
                    xlim = nombreMvt * 2 * 3
                if (Qte * 2) != ylim :
                    ylim = Qte * 2 * 3

            df = Df.copy()
            df["Code_article"] = ""
            df["Libellé_article"] = ""
            for i in df.index:
                df["Code_article"][i] = i[0]
                df["Libellé_article"][i] = i[1]

            for i in Df.index:
                if Df["Quantite"][i] < 0:
                    Df["Quantite"][i] = -1 * Df["Quantite"][i]
                    df["Quantite"][i] = -1 * df["Quantite"][i]


                if Df["Quantite"][i] > ylim:
                    Df["Quantite"][i] = ylim

                if Df["Nombre de mouvement"][i] > xlim:
                    Df["Nombre de mouvement"][i] = xlim
                for j in Df.index:
                    if Df["Quantite"][i] == Df["Quantite"][j] and Df["Nombre de mouvement"][i] == Df["Nombre de mouvement"][j]:
                        Df["Facteur"][i] += 1

            Df = Df.drop_duplicates()



            fig, ax = plt.subplots()

            for x in Df.index:
                ax.scatter(Df["Nombre de mouvement"][x],
                           Df["Quantite"][x],
                           s=200 * Df["Facteur"][x],
                           c="#004DFF",edgecolors = "black")

            ax.set_facecolor("#ADD8E6")

            fig.set_size_inches(18.5, 10.5)

            plt.xlabel("Mouvement")
            plt.ylabel("Quantités")


            #region Ajouter la légende ( les noms des catégories)

            plt.text(xlim/2 * 1.45, ylim/2 * 1.5,
                     "A", fontsize=30, color="black", bbox=dict(boxstyle="square",
                                                                        ec=(1., 0.5, 0.5),
                                                                        fc="none",
                                                                        ))
            plt.text(xlim/2 * 0.45, ylim/2 * 1.5,
                     "B", fontsize=30, color="black", bbox=dict(boxstyle="square",
                                                                        ec=(1, 0.5, 0.5),
                                                                        fc="none",
                                                                        ))
            plt.text(xlim/2 * 1.45, ylim/2 * 0.5,
                     "C", fontsize=30, color="black", bbox=dict(boxstyle="square",
                                                                        ec=(1., 0.5, 0.5),
                                                                        fc="None",
                                                                        ))
            plt.text(xlim / 2 * 0.45, ylim/2 * 0.5,
                     "D", fontsize=30, color="black", bbox=dict(boxstyle="square",
                                                                        ec=(1., 0.5, 0.5),
                                                                        fc="None",
                                                                        ))
            #endregion

            plt.xlim((0, xlim))
            plt.ylim((0, ylim))

            plt.vlines(xlim/2, 0, Df['Quantite'].max()*2, linestyles="solid", colors="k")

            plt.hlines(ylim/2, 0, Df['Quantite'].max(), linestyles="solid", colors="k")





            if str(startMonth) in  ["1","2","3","4","5","6","7","8","9"]:
                startMonth = str(0) + str(startMonth)
            if str(endMonth) in  ["1","2","3","4","5","6","7","8","9"]:
                endMonth = str(0) + str(endMonth)

            DateDebut = str(startMonth) + "/" + str(startYear)
            DateFin =  str(endMonth) + "/" + str(endYear)
            NombreArticleTotal = len(df.Code_article.unique().tolist())
            Titre1 = "#### Catégorisation sur les 3 derniers mois incluant " + str(NombreArticleTotal) + " articles" + " ( du "+ DateDebut + " au " + DateFin + " )"
            st.markdown(Titre1)
            plt.tight_layout()
            st.write(fig)



            #region Details des catégories

            DataframeA = df[np.logical_and((df["Quantite"] >= ylim / 2), (df["Nombre de mouvement"] >= xlim / 2))]
            DataframeA["Catégorie"] = "A"
            DataframeB = df[np.logical_and((df["Quantite"] >= ylim / 2), (df["Nombre de mouvement"] < xlim / 2))]
            DataframeB["Catégorie"] = "B"
            DataframeC = df[np.logical_and((df["Quantite"] < ylim / 2), (df["Nombre de mouvement"] >= xlim / 2))]
            DataframeC["Catégorie"] = "C"
            DataframeD = df[np.logical_and((df["Quantite"] < ylim / 2), (df["Nombre de mouvement"] < xlim / 2))]
            DataframeD["Catégorie"] = "D"
            IndexList = ["Référence", "Quantité", "Mouvement", "Ratio"]

            #region Categorie A

            NombreA = len(DataframeA.Code_article.unique().tolist())
            PourcentageNombreA = str(round((NombreA / NombreArticleTotal) * 100, 2)) + " %"

            QuantiteA = DataframeA["Quantite"].sum()
            PourcentageQuantiteA = str(round((QuantiteA / df["Quantite"].sum()) * 100, 2)) + " %"

            MvtA = DataframeA["Nombre de mouvement"].sum()
            PourcentageMvtA = str(round((MvtA / df["Nombre de mouvement"].sum()) * 100, 2)) + " %"

            RatioA = str(round((QuantiteA/MvtA))) + " vol/mvt"

            NombreA = str(len(DataframeA.Code_article.unique().tolist())) + " réf."
            QuantiteA = str(DataframeA["Quantite"].sum()) + " vol."
            MvtA = str(DataframeA["Nombre de mouvement"].sum()) + " mvt(s)"


            NombreList = [NombreA, QuantiteA, MvtA, RatioA]
            PourcentageList = [PourcentageNombreA, PourcentageQuantiteA, PourcentageMvtA, '']

            IndicateursADF = pd.DataFrame(list(zip(IndexList, NombreList, PourcentageList)),
                                         columns=["" ,"Nombre", "Pourcentage"])
            IndicateursADF.set_index("")

            #endregion

            #region Categorie B

            NombreB = len(DataframeB.Code_article.unique().tolist())
            PourcentageNombreB = str(round((NombreB / NombreArticleTotal) * 100, 2)) + " %"

            QuantiteB = DataframeB["Quantite"].sum()
            PourcentageQuantiteB = str(round((QuantiteB / df["Quantite"].sum()) * 100, 2)) + " %"

            MvtB = DataframeB["Nombre de mouvement"].sum()
            PourcentageMvtB = str(round((MvtB / df["Nombre de mouvement"].sum()) * 100, 2)) + " %"

            RatioB = str(round((QuantiteB/MvtB),2)) + " vol/mvt"

            NombreB = str(len(DataframeB.Code_article.unique().tolist())) + " réf."
            QuantiteB = str(DataframeB["Quantite"].sum()) + " vol."
            MvtB = str(DataframeB["Nombre de mouvement"].sum()) + " mvt(s)"

            NombreListB = [NombreB, QuantiteB, MvtB, RatioB]
            PourcentageListB = [PourcentageNombreB, PourcentageQuantiteB, PourcentageMvtB, '']

            IndicateursBDF = pd.DataFrame(list(zip(IndexList, NombreListB, PourcentageListB)),
                                          columns=[ "", "Nombre", "Pourcentage"])
            IndicateursBDF.set_index("")
            #endregion

            #region Categorie C

            NombreC =len(DataframeC.Code_article.unique().tolist())
            PourcentageNombreC = str(round((NombreC / NombreArticleTotal) * 100, 2)) + " %"

            QuantiteC = DataframeC["Quantite"].sum()
            PourcentageQuantiteC = str(round((QuantiteC / df["Quantite"].sum()) * 100, 2)) + " %"

            MvtC = DataframeC["Nombre de mouvement"].sum()
            PourcentageMvtC = str(round((MvtC / df["Nombre de mouvement"].sum()) * 100, 2)) + " %"

            RatioC = str(round((QuantiteC/MvtC),2)) + " vol/mvt"

            NombreC = str(len(DataframeC.Code_article.unique().tolist())) + " réf."
            QuantiteC = str(DataframeC["Quantite"].sum()) + " vol."
            MvtC = str(DataframeC["Nombre de mouvement"].sum()) + " mvt(s)"

            NombreListC = [NombreC, QuantiteC, MvtC, RatioC]
            PourcentageListC = [PourcentageNombreC, PourcentageQuantiteC, PourcentageMvtC, '']

            IndicateursCDF = pd.DataFrame(list(zip(IndexList, NombreListC, PourcentageListC)),
                                          columns=["", "Nombre", "Pourcentage"])
            IndicateursCDF.set_index("")

            #endregion

            #region Categorie D

            NombreD = len(DataframeD.Code_article.unique().tolist())
            PourcentageNombreD = str(round((NombreD / NombreArticleTotal) * 100, 2)) + " %"

            QuantiteD = DataframeD["Quantite"].sum()
            PourcentageQuantiteD = str(round((QuantiteD / df["Quantite"].sum()) * 100, 2)) + " %"

            MvtD = DataframeD["Nombre de mouvement"].sum()
            PourcentageMvtD = str(round((MvtD / df["Nombre de mouvement"].sum()) * 100, 2)) + " %"

            RatioD = str(round((QuantiteD/MvtD),2)) + " vol/mvt"

            NombreD = str(len(DataframeD.Code_article.unique().tolist())) + " réf."
            QuantiteD = str(DataframeD["Quantite"].sum()) + " vol."
            MvtD = str(DataframeD["Nombre de mouvement"].sum()) + " mvt(s)"

            NombreListD = [NombreD, QuantiteD, MvtD, RatioD]
            PourcentageListD = [PourcentageNombreD, PourcentageQuantiteD, PourcentageMvtD, '']

            IndicateursDDF = pd.DataFrame(list(zip( IndexList, NombreListD, PourcentageListD)),
                                          columns=["","Nombre", "Pourcentage"])
            IndicateursDDF.set_index("")
            #endregion



            #region Dataframe des catégories
            Mvts = Mouvements.copy()
            Mvts = Mvts[Mvts["Libellé_mouvement"] == "Expédition destinataire"]
            xlimite = nombreMvt
            ylimite = Qte

            ListeDates = []

            datesFin = Mvts.Date_création.max()
            for i in range(3):
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


                DfA = df[np.logical_and((df["Quantite"] >= ylimite ), (df["Nombre de mouvement"] >= xlimite ))]

                DfB = df[np.logical_and((df["Quantite"] >= ylimite ), (df["Nombre de mouvement"] < xlimite ))]

                DfC = df[np.logical_and((df["Quantite"] < ylimite), (df["Nombre de mouvement"] >= xlimite ))]

                DfD = df[np.logical_and((df["Quantite"] < ylimite ), (df["Nombre de mouvement"] < xlimite ))]
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

            #endregion


            DataframeExported = pd.DataFrame()
            DataframeExported = DataframeExported.append(DataframeA)
            DataframeExported = DataframeExported.append(DataframeB)
            DataframeExported = DataframeExported.append(DataframeC)
            DataframeExported = DataframeExported.append(DataframeD)
            DataframeExported = DataframeExported[["Code_article","Libellé_article","Catégorie"]]
            DataframeExported = DataframeExported.rename(columns = {"Catégorie" : "Catégorie sur la période donnée"})

            categorisationDf = pd.concat(dfs, axis=1)
            categorisationDf.fillna("D", inplace=True)
            categorisationDf.drop("Code_article", axis=1, inplace=True)
            categorisationDf["Code_article"] = categorisationDf.index
            categorisationDf.drop_duplicates()
            DataframeExported = DataframeExported.reset_index(drop=True)

            DfFinale = pd.merge(DataframeExported, categorisationDf, on="Code_article")


            save_path = "Catégorisation " + str(3) + " Mois.xlsx"
            df_xlsx = to_excel(DfFinale)
            st.download_button(label='📥 Télécharger ce résultat',
                               data=df_xlsx,
                               file_name=save_path)



            col1,col2= st.columns(2)
            with col1:
                st.markdown("<h5 style='text-align: center; color: red;'>Catégorie A</h5>", unsafe_allow_html=True)
                st.table(IndicateursADF.style.set_table_styles([{'selector': 'th',
                                                                        "props": [("color", "black"),("font-weight", "bold"), ("font-size", "20px")]}
                                                                   ,{"selector": "th.row_heading","props": [ ("color", "black"),("font-weight", "bold"),("font-size", "18px"),("text-align", "center") ] },
                                                                {"selector": "td","props": [("font-size", "18px") ]},
                                                                {"selector": "tbody>tr>:nth-child(2)","props": [("font-weight", "bold") ]}]))
                st.markdown("<h5 style='text-align: center; color: red;'>Catégorie C</h5>", unsafe_allow_html=True)
                st.table(IndicateursCDF.style.set_table_styles([{'selector': 'th',
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "20px")]},
                                                                {"selector": "th.row_heading",
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "18px"),
                                                                           ("text-align", "center")]},
                                                                {"selector": "td",
                                                                 "props": [("font-size", "18px")]},
                                                                {"selector": "tbody>tr>:nth-child(2)","props": [("font-weight", "bold") ]}]))
            with col2:
                st.markdown("<h5 style='text-align: center; color: red;'>Catégorie B</h5>", unsafe_allow_html=True)
                st.table(IndicateursBDF.style.set_table_styles([{'selector': 'th',
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "20px")]},
                                                                {"selector": "th.row_heading",
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "18px"),
                                                                           ("text-align", "center")]},
                                                                {"selector": "td",
                                                                 "props": [("font-size", "18px")]},
                                                                {"selector": "tbody>tr>:nth-child(2)","props": [("font-weight", "bold") ]}]))
                st.markdown("<h5 style='text-align: center; color: red;'>Catégorie D</h5>", unsafe_allow_html=True)
                st.table(IndicateursDDF.style.set_table_styles([{'selector': 'th',
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "20px")]},
                                                                {"selector": "th.row_heading",
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "18px"),
                                                                           ("text-align", "center")]},
                                                                {"selector": "td",
                                                                 "props": [("font-size", "18px")]},{"selector": "tbody>tr>:nth-child(2)","props": [("font-weight", "bold") ]}]))


            #endregion




            #endregion









            #region Magic Quadrant avec differentes périodes

            OptionDate = st.selectbox("Choisir un intervalle de mois", [6,9,12])


            startDate = setDateDebutWithMonth(OptionDate,Mouvements.Date_création.max())
            MoisDebut = startDate.month
            AnneeDebut = startDate.year

            dataframeMonth = Mouvements[Mouvements["Date_création"].isin(pd.date_range(start=startDate, end=endDate))]
            dataframeMonth = dataframeMonth[dataframeMonth["Libellé_mouvement"] == "Expédition destinataire"]
            ArticleQuantite = dataframeMonth.groupby(["Code_article","Libellé_article"])["Quantité"].apply(lambda x: x.sum())
            ArticleMouvement = dataframeMonth[["Code_article","Libellé_article"]].value_counts()
            Dframe = ArticleQuantite.to_frame().join(ArticleMouvement.to_frame())
            Dframe.columns = ["Quantite", "Nombre de mouvement"]


            dframe = Dframe.copy()
            if str(MoisDebut) in ["1","2","3","4","5","6","7","8","9"]:
                MoisDebut =  str(0) + str(MoisDebut)

            DebutDate = str(MoisDebut) + "/" + str(AnneeDebut)
            dframe["Code_article"] = ""
            dframe["Libellé_article"] = ""
            for i in dframe.index:
                dframe["Code_article"][i] = i[0]
                dframe["Libellé_article"][i] = i[1]
            NombreArticleTotal = len(dframe["Code_article"].unique().tolist())
            Titre2 = "#### Catégorisation sur les " + str(OptionDate) + " derniers mois incluant " + str(NombreArticleTotal) + " articles" + " ( du "+ DebutDate + " au " + DateFin + " )"
            st.markdown(Titre2)

            xintersection =  nombreMvt * OptionDate
            yintersection = Qte * OptionDate

            print(xintersection)
            print(yintersection)





            Dframe["Facteur"] = 0
            for i in Dframe.index :
                if Dframe["Quantite"][i] > yintersection * 2:
                    Dframe["Quantite"][i] = yintersection * 2

                if Dframe["Nombre de mouvement"][i] > xintersection*2 :
                    Dframe["Nombre de mouvement"][i] = xintersection * 2

                for j in Dframe.index :
                    if Dframe["Quantite"][i] == Dframe["Quantite"][j] and Dframe["Nombre de mouvement"][i] == Dframe["Nombre de mouvement"][j]:
                        Dframe["Facteur"][i] += 1

            Dframe = Dframe.drop_duplicates()

            #region Scatter Plot

            fig, ax = plt.subplots()
            for x in Dframe.index :
                ax.scatter ( Dframe["Nombre de mouvement"][x],
                             Dframe["Quantite"][x],
                             s = 200 * Dframe["Facteur"][x],
                             c = "#004DFF",edgecolors = "black")

            ax.set_facecolor("#ADD8E6")

            fig.set_size_inches(18.5, 10.5)

            plt.xlabel("Mouvement")
            plt.ylabel("Quantités")

            plt.text( xintersection * 1.45, yintersection * 1.5, "A", fontsize=30, color="black",
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc="none",
                               ))
            plt.text(xintersection * 0.45, yintersection * 1.5, "B", fontsize=30, color="black",
                     bbox=dict(boxstyle="square",
                               ec=(1, 0.5, 0.5),
                               fc="none",
                               ))
            plt.text(xintersection * 1.45, yintersection * 0.5, "C", fontsize=30, color="black",
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc="None",
                               ))
            plt.text(xintersection * 0.45, yintersection * 0.5, "D", fontsize=30, color="black",
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc="None",
                               ))


            plt.xticks(np.arange(0, xintersection * 2, round(xintersection * 2 / 8)))
            plt.yticks(np.arange(0, yintersection * 2, round(yintersection *2 /9 ) ))

            plt.xlim((0, xintersection * 2))
            plt.ylim((0, yintersection * 2))

            plt.vlines(xintersection, 0, yintersection * 2.5 , linestyles="solid", colors="k")
            plt.hlines(yintersection, 0, xintersection * 2.5, linestyles="solid", colors="k")

            plt.tight_layout()

            st.write(fig)
            #endregion




            #region Catégorisation mois par mois
            Mvts = Mouvements.copy()
            Mvts = Mvts[Mvts["Libellé_mouvement"] == "Expédition destinataire"]


            listeDate = []
            dateFin = Mouvements.Date_création.max()
            for i in range(OptionDate):
                fin = dateFin
                start = setDateDebutWithMonth(1,fin)
                dateFin = setDateFin(start)
                listeDate.append((fin, start))
            listeDate.reverse()
            dframes = []
            for periode in listeDate :


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

                DfA = df[np.logical_and((df["Quantite"] >= Qte), (df["Nombre de mouvement"] >= nombreMvt))]
                DfB = df[np.logical_and((df["Quantite"] >= Qte), (df["Nombre de mouvement"] < nombreMvt))]
                DfC = df[np.logical_and((df["Quantite"] < Qte), (df["Nombre de mouvement"] >= nombreMvt))]
                DfD = df[np.logical_and((df["Quantite"] < Qte), (df["Nombre de mouvement"] < nombreMvt))]
                column_name = str(endDate.month) + "/" + str(endDate.year)
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
                dframes.append(DfExported)








            #endregion



            # region Details des catégories

            DframeA = dframe[np.logical_and((dframe["Quantite"] >= yintersection ), (dframe["Nombre de mouvement"] >= xintersection ))]
            DframeB = dframe[np.logical_and((dframe["Quantite"] >= yintersection ), (dframe["Nombre de mouvement"] < xintersection))]
            DframeC = dframe[np.logical_and((dframe["Quantite"] < yintersection ), (dframe["Nombre de mouvement"] >= xintersection ))]
            DframeD = dframe[np.logical_and((dframe["Quantite"] < yintersection ), (dframe["Nombre de mouvement"] < xintersection ))]


            DframeA["Catégorie"] = "A"
            DframeB["Catégorie"] = "B"
            DframeC["Catégorie"] = "C"
            DframeD["Catégorie"] = "D"

            IndexList = ["Référence", "Quantité", "Mouvement", "Ratio"]

            # region Categorie A

            NombreA = len(DframeA.Code_article.unique().tolist())
            PourcentageNombreA = str(round((NombreA / NombreArticleTotal) * 100, 2)) + " %"

            QuantiteA = DframeA["Quantite"].sum()
            PourcentageQuantiteA = str(round((QuantiteA / dframe["Quantite"].sum()) * 100, 2)) + " %"

            MvtA = DframeA["Nombre de mouvement"].sum()
            PourcentageMvtA = str(round((MvtA / dframe["Nombre de mouvement"].sum()) * 100, 2)) + " %"

            RatioA = str(round((QuantiteA / MvtA))) + " vol/mvt"

            NombreA = str(len(DframeA.Code_article.unique().tolist())) + " réf."
            QuantiteA = str(DframeA["Quantite"].sum()) + " vol."
            MvtA = str(DframeA["Nombre de mouvement"].sum()) + " mvt(s)"

            NombreList = [NombreA, QuantiteA, MvtA, RatioA]
            PourcentageList = [PourcentageNombreA, PourcentageQuantiteA, PourcentageMvtA, '']

            IndicateursADframe = pd.DataFrame(list(zip(IndexList, NombreList, PourcentageList)),
                                          columns=["", "Nombre", "Pourcentage"])
            IndicateursADframe.set_index("")

            # endregion

            # region Categorie B

            NombreB = len(DframeB.Code_article.unique().tolist())
            PourcentageNombreB = str(round((NombreB / NombreArticleTotal) * 100, 2)) + " %"

            QuantiteB = DframeB["Quantite"].sum()
            PourcentageQuantiteB = str(round((QuantiteB / dframe["Quantite"].sum()) * 100, 2)) + " %"

            MvtB = DframeB["Nombre de mouvement"].sum()
            PourcentageMvtB = str(round((MvtB / dframe["Nombre de mouvement"].sum()) * 100, 2)) + " %"

            RatioB = str(round((QuantiteB / MvtB), 2)) + " vol/mvt"

            NombreB = str(len(DframeB.Code_article.unique().tolist())) + " réf."
            QuantiteB = str(DframeB["Quantite"].sum()) + " vol."
            MvtB = str(DframeB["Nombre de mouvement"].sum()) + " mvt(s)"

            NombreListB = [NombreB, QuantiteB, MvtB, RatioB]
            PourcentageListB = [PourcentageNombreB, PourcentageQuantiteB, PourcentageMvtB, '']

            IndicateursBDframe = pd.DataFrame(list(zip(IndexList, NombreListB, PourcentageListB)),
                                          columns=["", "Nombre", "Pourcentage"])
            IndicateursBDframe.set_index("")
            # endregion

            # region Categorie C

            NombreC = len(DframeC.Code_article.unique().tolist())
            PourcentageNombreC = str(round((NombreC / NombreArticleTotal) * 100, 2)) + " %"

            QuantiteC = DframeC["Quantite"].sum()
            PourcentageQuantiteC = str(round((QuantiteC / dframe["Quantite"].sum()) * 100, 2)) + " %"

            MvtC = DframeC["Nombre de mouvement"].sum()
            PourcentageMvtC = str(round((MvtC / dframe["Nombre de mouvement"].sum()) * 100, 2)) + " %"

            RatioC = str(round((QuantiteC / MvtC), 2)) + " vol/mvt"

            NombreC = str(len(DframeC.Code_article.unique().tolist())) + " réf."
            QuantiteC = str(DframeC["Quantite"].sum()) + " vol."
            MvtC = str(DframeC["Nombre de mouvement"].sum()) + " mvt(s)"

            NombreListC = [NombreC, QuantiteC, MvtC, RatioC]
            PourcentageListC = [PourcentageNombreC, PourcentageQuantiteC, PourcentageMvtC, '']

            IndicateursCDframe = pd.DataFrame(list(zip(IndexList, NombreListC, PourcentageListC)),
                                          columns=["", "Nombre", "Pourcentage"])
            IndicateursCDframe.set_index("")

            # endregion

            # region Categorie D

            NombreD = len(DframeD.Code_article.unique().tolist())
            PourcentageNombreD = str(round((NombreD / NombreArticleTotal) * 100, 2)) + " %"

            QuantiteD = DframeD["Quantite"].sum()
            PourcentageQuantiteD = str(round((QuantiteD / dframe["Quantite"].sum()) * 100, 2)) + " %"

            MvtD = DataframeD["Nombre de mouvement"].sum()
            PourcentageMvtD = str(round((MvtD / dframe["Nombre de mouvement"].sum()) * 100, 2)) + " %"

            RatioD = str(round((QuantiteD / MvtD), 2)) + " vol/mvt"

            NombreD = str(len(DframeD.Code_article.unique().tolist())) + " réf."
            QuantiteD = str(DframeD["Quantite"].sum()) + " vol."
            MvtD = str(DframeD["Nombre de mouvement"].sum()) + " mvt(s)"

            NombreListD = [NombreD, QuantiteD, MvtD, RatioD]
            PourcentageListD = [PourcentageNombreD, PourcentageQuantiteD, PourcentageMvtD, '']

            IndicateursDDframe = pd.DataFrame(list(zip(IndexList, NombreListD, PourcentageListD)),
                                          columns=["", "Nombre", "Pourcentage"])
            IndicateursDDframe.set_index("")
            # endregion

            DframeExported = pd.DataFrame()
            DframeExported = DframeExported.append(DframeA)
            DframeExported = DframeExported.append(DframeB)
            DframeExported = DframeExported.append(DframeC)
            DframeExported = DframeExported.append(DframeD)

            DframeExported = DframeExported[["Code_article", "Libellé_article","Catégorie"]]
            DframeExported = DframeExported.rename(columns={"Catégorie" : "Catégorie sur la période donnée"})

            categorisationDframe = pd.concat(dframes, axis=1)
            categorisationDframe.fillna("D", inplace=True)
            categorisationDframe.drop("Code_article", axis=1, inplace=True)
            categorisationDframe["Code_article"] = categorisationDframe.index
            categorisationDframe.drop_duplicates()
            DframeExported = DframeExported.reset_index(drop=True)
            DframeFinale = pd.merge(DframeExported, categorisationDframe, on="Code_article")

            saving_path = "Catégorisation " + str(OptionDate) + " Mois.xlsx"
            dframe_xlsx = to_excel(DframeFinale)
            st.download_button(label='📥 Télécharger ce résultat',
                               data=dframe_xlsx,
                               file_name=saving_path)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h5 style='text-align: center; color: red;'>Catégorie A</h5>", unsafe_allow_html=True)
                st.table(IndicateursADframe.style.set_table_styles([{'selector': 'th',
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "20px")]}
                                                                   , {"selector": "th.row_heading",
                                                                      "props": [("color", "black"),
                                                                                ("font-weight", "bold"),
                                                                                ("font-size", "18px"),
                                                                                ("text-align", "center")]},
                                                                {"selector": "td", "props": [("font-size", "18px")]},
                                                                {"selector": "tbody>tr>:nth-child(2)",
                                                                 "props": [("font-weight", "bold")]}]))
                st.markdown("<h5 style='text-align: center; color: red;'>Catégorie C</h5>", unsafe_allow_html=True)
                st.table(IndicateursCDframe.style.set_table_styles([{'selector': 'th',
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "20px")]},
                                                                {"selector": "th.row_heading",
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "18px"),
                                                                           ("text-align", "center")]},
                                                                {"selector": "td",
                                                                 "props": [("font-size", "18px")]},
                                                                {"selector": "tbody>tr>:nth-child(2)",
                                                                 "props": [("font-weight", "bold")]}]))
            with col2:
                st.markdown("<h5 style='text-align: center; color: red;'>Catégorie B</h5>", unsafe_allow_html=True)
                st.table(IndicateursBDframe.style.set_table_styles([{'selector': 'th',
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "20px")]},
                                                                {"selector": "th.row_heading",
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "18px"),
                                                                           ("text-align", "center")]},
                                                                {"selector": "td",
                                                                 "props": [("font-size", "18px")]},
                                                                {"selector": "tbody>tr>:nth-child(2)",
                                                                 "props": [("font-weight", "bold")]}]))
                st.markdown("<h5 style='text-align: center; color: red;'>Catégorie D</h5>", unsafe_allow_html=True)
                st.table(IndicateursDDframe.style.set_table_styles([{'selector': 'th',
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "20px")]},
                                                                {"selector": "th.row_heading",
                                                                 "props": [("color", "black"), ("font-weight", "bold"),
                                                                           ("font-size", "18px"),
                                                                           ("text-align", "center")]},
                                                                {"selector": "td",
                                                                 "props": [("font-size", "18px")]},
                                                                {"selector": "tbody>tr>:nth-child(2)",
                                                                 "props": [("font-weight", "bold")]}]))

            # endregion



            #endregion

















