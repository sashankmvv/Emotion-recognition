import matplotlib.pyplot as plt
from utils import *
from model import get_model

basic_emotions = ['surprise', 'fear', 'disgust',
                  'happy', 'sad', 'angry', 'neutral']

# This dictionary can be used to interpret the output in form of its actual labels
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


def plot_image(input):
    model = get_model(pretrained=True, dataset_name="RAFDB")
    inp = np.expand_dims(input, axis=0)
    prediction = model.predict(inp)

    plt.figure(figsize=(5, 5))
    plt.imshow(input/255.0, cmap='gray')
    plt.axis('off')
    txt = f"Emotion : {dataset_dict['emotion_id'][np.argmax(prediction[0])]} \nGender : {dataset_dict['gender_id'][np.argmax(prediction[1])]}\nRace : {dataset_dict['race_id'][np.argmax(prediction[2])]}\nAge : {dataset_dict['age_id'][np.argmax(prediction[3])]}"
    plt.figtext(0.5, 0.01, txt, wrap=True,
                horizontalalignment='center', fontsize=11, fontweight="bold", color="white", backgroundcolor="black")


X, y = load_data("RAFDB")

print("Press Control+C to stop visualizing plots")
while True:
    i = np.random.randint(0, len(X))
    plot_image(X[i])
    plt.show()
