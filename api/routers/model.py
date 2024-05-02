from fastapi import APIRouter
import pandas as pd
from faker import Faker
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


model = APIRouter(prefix="/model")


def load_data() -> pd.DataFrame:
    # Carregar os dados do request (exemplo)
    df = pd.read_csv("api/dataset/name_gender_dataset.csv")

    # Converter os dados para um DataFrame (exemplo)
    df = df.loc[:, ['Name', 'Gender']]

    return df


def preprocess_data(dataframe: pd.DataFrame):
    df = load_data()

    masc_names = df[df['Gender'] == 'M'].shape[0]
    fem_names = df[df['Gender'] == 'F'].shape[0]

    needed_number_of_names = abs(masc_names - fem_names)

    faker = Faker()
    generated_names = []

    if masc_names < fem_names:
        name = faker.first_name_male()
        sex = 'M'

    elif fem_names < masc_names:
        name = faker.first_name_fem()
        sex = 'F'

    for _ in range(needed_number_of_names):
        generated_names.append([name, sex])

    generated_names_df = pd.DataFrame(
        generated_names, columns=['Name', 'Gender'])

    df = pd.concat([df, generated_names_df], ignore_index=True)

    df['Gender'].replace({'F': 0, 'M': 1}, inplace=True)


    names = df['Name'].values  # esse vetor é considerado um documento de texto

    vectorizer = CountVectorizer(analyzer='char')
    count_matrix = vectorizer.fit_transform(names)
    count_array = count_matrix.toarray()

    caracteres = vectorizer.get_feature_names_out()
    count_letter_df = pd.DataFrame(data=count_array, columns=caracteres, index=names)
    
    X = count_letter_df

    # Label
    y = df['Gender']

    return X, y


@model.get("/train-model")
async def train_model():
    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=45)

    classifier = KNeighborsClassifier(n_neighbors=50)
    classifier.fit(X_train.values, y_train.values)

    y_pred = classifier.predict(X_test.values)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
