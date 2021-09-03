import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve


def open_pickle(url):
    infile = urlretrieve(url, 'final_x.pkl')
    # X_dict = infile.read()
    # infile.close()


X_dict = open_pickle(
    "https: // drive.google.com/file/d/1nAjHb2SioYCmEGsSW52t1p8iWcU_Yk6W/view?usp=sharing")
y = open_pickle(
    "https: // drive.google.com/file/d/1nAjHb2SioYCmEGsSW52t1p8iWcU_Yk6W/view?usp=sharing")
X = np.zeros((X_dict.shape[0], 50, 50, 3))
for i in range(X.shape[0]):
    data_img = X_dict[i].copy()
    data_img = cv2.merge((data_img, data_img, data_img))
    #data_img = cv2.resize(data_img, (224, 224))
    X[i] = data_img
del X_dict
basic_emotions = ['surprise', 'fear', 'disgust',
                  'happy', 'sad', 'angry', 'neutral']
IMG_WIDTH, IMG_HEIGHT = 50, 50

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


for k in dataset_dict.keys():
    if "alias" in k:
        print(f"{k} : {dataset_dict[k]}")

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, shuffle=True, random_state=69)

del X


def seperatecategory(y_train):
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
