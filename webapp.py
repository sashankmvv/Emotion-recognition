import streamlit as st
from model import get_model
import numpy as np
from PIL import Image
import cv2


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = get_model(pretrained=True, dataset_name="RAFDB")
basic_emotions = ['surprise', 'fear', 'disgust',
                  'happy', 'sad', 'angry', 'neutral']


def detect_faces(our_image):
    img = np.array(our_image)
    # Detect faces (not able to detect faces for the images which contains face image only)
    # faces = face_cascade.detectMultiScale(img, 1.1, 4)
    # # Draw rectangle around the faces

    # for (x, y, w, h) in faces:
    # To draw a rectangle in a face
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    # face = img[y:y+h, x:x+w]
    face_resized = cv2.resize(img, (50, 50))
    try:
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    except:
        gray = face_resized.copy()
    gray_three = cv2.merge([gray, gray, gray])
    inp = np.expand_dims(gray_three, axis=0)
    preds = model.predict(inp)  # predicts [emotion, gender, race, age]
    pred = preds[0]
    emotion = basic_emotions[np.argmax(pred)]
    # cv2.putText(img, emotion, (20, 0 + 20 ),
    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 0, 255))

    return emotion


def main():

    html_temp = """
    <body>
    <div style="padding:10px">
    <h2 style="text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        data_load_state = st.text('Predicting...')

        with open("style.css") as f:
            st.markdown('<style>{}</style>'.format(f.read()),
                        unsafe_allow_html=True)
        our_image = Image.open(image_file)
        st.image(our_image, width=250)
        emotion = detect_faces(our_image)
        st.write(f"Prediction : {emotion}")

        data_load_state.text("")


if __name__ == '__main__':
    main()
