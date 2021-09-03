import numpy as np
import cv2
import pickle
import gdown

# RAFDB_X_URL = "https://drive.google.com/file/d/1nAjHb2SioYCmEGsSW52t1p8iWcU_Yk6W/view?usp=sharing"
# RAFDB_y_URL = "https://drive.google.com/file/d/19YAXA3Pu4U32W8h7fM6wUEwH-5oeQuPl/view?usp=sharing"

RAFDB_X_URL = "https://drive.google.com/uc?id=1nAjHb2SioYCmEGsSW52t1p8iWcU_Yk6W"
RAFDB_y_URL = "https://drive.google.com/uc?id=19YAXA3Pu4U32W8h7fM6wUEwH-5oeQuPl"

FER_X_URL = "https://drive.google.com/uc?id=1KMDivwGo8XGTpMOgQ83LC0tdvOv2vOzH"
FER_y_URL = "https://drive.google.com/uc?id=1jNQKvnrRcARieeP83jJpRNGh_XSX0Xtj"


def download_data(dataset_name):
    output_X = f"./{dataset_name}_X.pkl"
    output_y = f"./{dataset_name}_y.pkl"

    print(f"Downloading {dataset_name} dataset ...")
    if dataset_name == "RAFDB":
        gdown.download(RAFDB_X_URL, output_X, quiet=False)
        gdown.download(RAFDB_y_URL, output_y, quiet=False)
    elif dataset_name == "FER":
        gdown.download(FER_X_URL, output_X, quiet=False)
        gdown.download(FER_y_URL, output_y, quiet=False)
    else:
        raise Exception(f"Invalid dataset name : {dataset_name}")
    return output_X, output_y


def expand_X_dim(X_dict):
    X = np.zeros((X_dict.shape[0], 50, 50, 3))
    for i in range(X.shape[0]):
        data_img = X_dict[i].copy()
        data_img = cv2.merge((data_img, data_img, data_img))
        X[i] = data_img
    del X_dict
    return X


def load_data(dataset_name):
    X_path, y_path = download_data(dataset_name)
    with open(X_path, "rb") as f:
        X = pickle.load(f)
    with open(y_path, "rb") as f:
        y = pickle.load(f)

    X = expand_X_dim(X)
    return X, y


dataset_dict = {
    'emotion_id': {
        0: "Surprise",
        1: "Fear",
        2: "Disgust",
        3: "Happiness",
        4: "Sadness",
        5: "Anger",
        6: "Neutral"
    },
    'gender_id': {
        0: 'male',
        1: 'female',
        2: 'unsure'
    },
    'race_id': {
        0: 'caucasian',
        1: 'African-American',
        2: 'Asian'
    },
    'age_id': {
        0: '0-3',
        1: '4-19',
        2: '20-39',
        3: '40-69',
        4: '70+'
    }
}
dataset_dict['emotion_alias'] = dict(
    (e, i) for i, e in dataset_dict['emotion_id'].items())
dataset_dict['gender_alias'] = dict(
    (g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((r, i)
                                  for i, r in dataset_dict['race_id'].items())
dataset_dict['age_alias'] = dict((a, i)
                                 for i, a in dataset_dict['age_id'].items())


# for k in dataset_dict.keys():
#     if "alias" in k:
#         print(f"{k} : {dataset_dict[k]}")


def seperate_category(y_train):
    ages_train, races_train, genders_train, emotions_train = [], [], [], []
    y_emotion = y_train[:, :7]
    y_gender = y_train[:, 7:10]
    y_race = y_train[:, 10:13]
    y_age = y_train[:, 13:]

    for i in range(y_emotion.shape[0]):
        ages_train.append(y_age[i])
        emotions_train.append(y_emotion[i])
        genders_train.append(y_gender[i])
        races_train.append(y_race[i])

    return ages_train, emotions_train, genders_train, races_train


def generate_images(X, y_emotion, y_gender, y_race, y_age, batch_size, is_training):
    images, ages, races, genders, emotions = [], [], [], [], []
    while True:
        for i in range(X.shape[0]):
            images.append(X[i])
            emotions.append(y_emotion[i])
            genders.append(y_gender[i])
            races.append(y_race[i])
            ages.append(y_age[i])
            if len(images) >= batch_size:
                yield np.array(images), [np.array(emotions), np.array(genders), np.array(races), np.array(ages)]
                images, emotions, ages, races, genders = [], [], [], [], []
        if not is_training:
            break
