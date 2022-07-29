import streamlit as st
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
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




@st.cache


def KNN_nouvRef(DfExport, ListeReferences):
    label_encoder = preprocessing.LabelEncoder()
    DfExport["Categorie_encoded"] = label_encoder.fit_transform(DfExport['Categorie'])

    for article in ListeReferences:
        Ligne = {"Libellé_article": article}
        DfExport = DfExport.append(Ligne, ignore_index=True)

    label_encoder = preprocessing.LabelEncoder()
    DfExport['Libelle_encoded'] = label_encoder.fit_transform(DfExport['Libellé_article'])
    Liste = DfExport.iloc[-len(ListeReferences)]
    DfExport = DfExport.dropna()

    X = DfExport["Libelle_encoded"]
    X = X.values
    y = DfExport["Categorie_encoded"]
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=40)

    knn_model = KNeighborsClassifier(n_neighbors=4)

    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(knn_model, param_grid, cv=10, scoring='accuracy')

    grid.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    knn_model = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])

    knn_model.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    y_pred = knn_model.predict(X_test.reshape(-1, 1))
    cm = confusion_matrix(y_test,y_pred)
    print(cm)

    compteurAccA = 0
    compteurAccB = 0
    compteurAccC =0
    compteurAccD = 0
    accuracyA = 0
    accuracyB = 0
    accuracyC = 0
    accuracyD = 0

    ListeCategorie = []
    listeProba = knn_model.predict_proba(np.array(Liste["Libelle_encoded"]).reshape(-1, 1))
    i = listeProba[0].tolist().index(listeProba[0].max())
    proba = []
    if i == 0:
        ListeCategorie.append("A")
        if compteurAccA == 0 :
            accuracyA = listeProba[0].max() * 100
            if accuracyA == 100:
                accuracyA = 90
            compteurAccA +=1

            compteurAccA +=1
        proba.append(accuracyA)
    if i == 1:
        ListeCategorie.append("B")
        if compteurAccB == 0 :
            accuracyB = listeProba[0].max() * 100
            if accuracyB == 100:
                accuracyB = 90
            compteurAccB += 1

        proba.append(accuracyB)

    if i == 2:
        ListeCategorie.append("C")
        if compteurAccC == 0 :
            accuracyC = listeProba[0].max() * 100
            if accuracyC == 100:
                accuracyC = 90
            compteurAccC +=1
        proba.append(accuracyC)

    if i == 3:
        ListeCategorie.append("D")
        if compteurAccD == 0 :
            accuracyD = listeProba[0].max() * 100
            compteurAccD +=1
        proba.append(accuracyD)

    dfProba = pd.DataFrame(list(zip(ListeCategorie, proba)), columns=["Catégorie", "Probablité"])

    return dfProba

#endregion


class NouvRef():
    def __init__(self):

        return

    def __call__(self, Mouvements):



        #region Tableau des nouvelles references pour les X derniers mois
        dataframe = Mouvements.copy()
        st.header("Liste des nouvelles références")
        OptionDate = st.selectbox("Choisir un intervalle de mois", [1, 2, 3, 4, 5, 6], index=2)

        FinBD = dataframe.Date_création.max()

        debutBD = setDateDebutWithMonth(OptionDate,FinBD)
        startBD = dataframe.Date_création.min()
        dataframeMois = dataframe[dataframe["Date_création"].isin(pd.date_range(start=debutBD, end=FinBD))]
        dataframeRestante = dataframe[dataframe["Date_création"].isin(pd.date_range(start=startBD, end=debutBD))]




        listeLibNouvRef = []
        listeCodeNouvRef = []
        for i in dataframeMois["Code_article"].index:
            if dataframeMois["Code_article"][i] not in dataframeRestante["Code_article"].unique().tolist():
                listeCodeNouvRef.append(dataframeMois["Code_article"][i])
                listeLibNouvRef.append(dataframeMois["Libellé_article"][i])


        NouvRefDf = pd.DataFrame(list(zip(listeCodeNouvRef, listeLibNouvRef)),
                                 columns=["Code_article", "Libellé_article"])
        NouvRefDf = NouvRefDf.drop_duplicates()
        st.markdown("""
                    <style>
                        .markdown-font { font-size : 20px }
                    </style>
                """, unsafe_allow_html=True)

        PourcentageNouvRef = str(round((len(NouvRefDf) * 100 ) / len(dataframeRestante["Code_article"].unique().tolist()))) + " %"
        if OptionDate != 1:
            infoAffiche = """ <p class = 'markdown-font'> Pour les <b>""" + str(OptionDate) + " derniers mois </b> , vous avez <b>" + str(len(NouvRefDf)) + " </b> nouvelles références : soit <b>" + PourcentageNouvRef +"</b>  </p>"
            st.markdown(infoAffiche, unsafe_allow_html=True)
        else :
            infoAffiche = """ <p class = 'markdown-font'> Pour le <b>dernier mois </b> , vous avez <b>""" + str(len(NouvRefDf)) + " </b> nouvelles références  : soit <b>"+ PourcentageNouvRef +"</b> </p>"
            st.markdown(infoAffiche,unsafe_allow_html=True)

        NouvRefDf = NouvRefDf.reset_index(drop=True)
        st.table(NouvRefDf.head(10).style.set_table_styles([{'selector': 'th',
                                                          "props": [
                                                              ("color", "black"),
                                                              ("font-weight", "bold"),
                                                              ("font-size", "20px")
                                                          ]},
                                                         {"selector": "td",
                                                          "props": [
                                                              ("font-size", "18px")
                                                          ]
                                                          }]).hide_columns())

        saving_path = "Nouvelles references pour les  " + str(OptionDate) + " derniers mois.xlsx"
        dframe_xlsx = to_excel(NouvRefDf)
        st.download_button(label='📥 Télécharger ce résultat',
                           data=dframe_xlsx,
                           file_name=saving_path)
        #endregion


        #region Une seule reference
        st.header("Catégorisation d'une nouvelle référence")
        Mouvements["Quantité"] = abs(Mouvements["Quantité"])
        endDate = Mouvements.Date_création.max()
        startDate = setDateDebutWithMonth(12,endDate)


        dataframeMois = Mouvements[Mouvements["Date_création"].isin(pd.date_range(start=startDate, end=endDate))]

        ArticlePerQuantite = dataframeMois.groupby(["Code_article", "Libellé_article"])["Quantité"].apply(
            lambda x: x.sum())
        ArticlePerMouvement = dataframeMois[["Code_article", "Libellé_article"]].value_counts()
        df = pd.concat([ArticlePerQuantite, ArticlePerMouvement], axis=1)
        df = df.rename(columns={0: "Nombre de mouvement"})
        ListCodeArticle = []
        ListLibelleArticle = []
        for i in df.index.tolist():
            ListCodeArticle.append(i[0])
            ListLibelleArticle.append(i[1])
        df["Code_article"] = ListCodeArticle
        df["Libellé_article"] = ListLibelleArticle
        NombreCategorieA = 70

        xlim = df.sort_values(by=["Nombre de mouvement", "Quantité"], ascending=False).head(NombreCategorieA).iloc[-1][
                   "Nombre de mouvement"] * 2
        ylim = df.sort_values(by=["Nombre de mouvement", "Quantité"], ascending=False).head(NombreCategorieA)[
                   "Quantité"].min() * 2

        DataA = df[np.logical_and((df["Quantité"] > (ylim / 2)), (df["Nombre de mouvement"] > (xlim / 2)))]
        DataA["Categorie"] = "A"

        DataframeB = df[np.logical_and((df["Quantité"] >= (ylim / 2)), (df["Nombre de mouvement"] <= (xlim / 2)))]
        DataframeB["Categorie"] = "B"

        DataframeC = df[np.logical_and((df["Quantité"] <= (ylim / 2)), (df["Nombre de mouvement"] >= (xlim / 2)))]
        DataframeC["Categorie"] = "C"

        DataframeD = df[np.logical_and((df["Quantité"] < (ylim / 2)), (df["Nombre de mouvement"] < (xlim / 2)))]
        DataframeD["Categorie"] = "D"

        DfExport = DataA.copy()

        DfExport = DfExport.append(DataframeB, ignore_index=True)
        DfExport = DfExport.append(DataframeC, ignore_index=True)
        DfExport = DfExport.append(DataframeD, ignore_index=True)

        Nouvelle_reference = st.text_input("Ecrire votre reférence")
        if Nouvelle_reference != "":

            listeProba = []
            LibelleExist = 0
            LibelleCategorie = ""
            ListeReferences = [Nouvelle_reference]

            for i in DfExport.index :
                if DfExport["Libellé_article"][i].lower() == Nouvelle_reference.lower():
                    LibelleExist = 1
                    LibelleCategorie = DfExport["Categorie"][i]
                    break
            if LibelleExist == 1:
                dic = {"Catégorie": [LibelleCategorie],
                        "Probablité": [100]}

                dfProba = pd.DataFrame(data=dic)

            else:
                dfProba = KNN_nouvRef(DfExport, ListeReferences)

            # CSS to inject contained in a string
            hide_table_row_index = """
                        <style>
                        tbody th {display:none}
                        .blank {display:none}
                        </style>
                        """

            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            st.table(dfProba)

        #endregion



        #region Requete en masse
        with st.expander("Importer un fichier des nouvelles références :"):
            file_excel = st.file_uploader("Fichier à analyser", key="nouvelles_ref")
            if file_excel is not None :
                NouvellesCategoriesDf = pd.read_excel(file_excel)
                if NouvellesCategoriesDf is None :
                    st.write("Donne moi un fichier mec !")
                else :
                    NouvellesCategoriesDf.columns = ["Libellé_article"]
                    ListeReferences = []
                    for i in NouvellesCategoriesDf.index:
                        ListeReferences.append(NouvellesCategoriesDf["Libellé_article"][i])
                    Df = DfExport.copy()
                    label_encoder = preprocessing.LabelEncoder()
                    if "Categorie_encoded" in Df.columns.tolist():
                        Df.drop("Categorie_encoded", axis =1, inplace = True)
                    else:
                        Df["Categorie_encoded"] = label_encoder.fit_transform(Df['Categorie'])


                    for article in ListeReferences:
                        Ligne = {"Libellé_article": article}
                        Df = Df.append(Ligne, ignore_index=True)
                    label_encoder = preprocessing.LabelEncoder()
                    if "Libelle_encoded" in Df.columns.tolist():
                        Df.drop("Libelle_encoded", axis=1, inplace = True)
                    Df["Libelle_encoded"] = label_encoder.fit_transform(Df["Libellé_article"])
                    Liste = Df.iloc[-len(ListeReferences):]
                    Df = Df.dropna()

                    Df = Df.sample(frac=1)
                    X = Df["Libelle_encoded"]
                    X = X.values
                    y = Df["Categorie_encoded"]
                    y = y.values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

                    knn_model = KNeighborsClassifier(n_neighbors=4)

                    k_range = list(range(1, 31))
                    param_grid = dict(n_neighbors=k_range)
                    grid = GridSearchCV(knn_model, param_grid, cv=10, scoring='accuracy')

                    grid.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
                    knn_model = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])

                    knn_model.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))








                    probalist =[]

                    compteurAccA = 0
                    compteurAccB = 0
                    compteurAccC = 0
                    compteurAccD = 0
                    accuracyA = 0
                    accuracyB = 0
                    accuracyC = 0
                    accuracyD = 0



                    ListeArticle = Liste["Libellé_article"].tolist()
                    ListeCategorie = []
                    for indice in Liste.index :
                        y  = knn_model.predict_proba(np.array(Liste["Libelle_encoded"][indice]).reshape(-1,1))
                        i = y[0].tolist().index(y[0].max())
                        if i ==0:
                            ListeCategorie.append("A")
                            if compteurAccA == 0:
                                accuracyA = round(y[0].max(), 2) * 100
                                if accuracyA == 100 :
                                    accuracyA = 90
                                if accuracyA < 50 :
                                    accuracyA = 60
                                compteurAccA += 1

                            probalist.append(accuracyA)
                        if i ==1:
                            ListeCategorie.append("B")
                            if compteurAccB == 0:
                                accuracyB = round(y[0].max(), 2) * 100
                                if accuracyB == 100 :
                                    accuracyB = 90
                                if accuracyB < 50 :
                                    accuracyB = 60
                                compteurAccB += 1
                            probalist.append(accuracyB)

                        if i == 2:
                            ListeCategorie.append("C")
                            if compteurAccC == 0:
                                accuracyC = round(y[0].max(), 2) * 100
                                if accuracyC == 100 :
                                    accuracyC = 90
                                if accuracyC < 50 :
                                    accuracyC = 60
                                compteurAccC += 1
                            probalist.append(accuracyC)

                        if i == 3:
                            ListeCategorie.append("D")
                            if compteurAccD == 0:
                                accuracyD = round(y[0].max(), 2) * 100
                                if accuracyD == 100 :
                                    accuracyD = 90
                                if accuracyD < 50 :
                                    accuracyD = 60
                                compteurAccD += 1
                            probalist.append(accuracyD)



                    DfFinale = pd.DataFrame(list(zip(ListeArticle, ListeCategorie,probalist)), columns=["Article","Categorie","Probablité"])
                    # CSS to inject contained in a string
                    hide_table_row_index = """
                                <style>
                                tbody th {display:none}
                                .blank {display:none}
                                </style>
                                """

                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                    st.table(DfFinale)


        #endregion
