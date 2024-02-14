from function import *
from keras.models import model_from_json
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3



engine = pyttsx3.init()
"""Voice Speed"""
rate = engine.getProperty('rate')  # getting details of current speaking rate
# print(rate)                        #printing current voice rate
engine.setProperty('rate', 155)  # setting up new voice rate

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

colors = []
for i in range(0, 20):
    colors.append((245, 117, 16))
print(len(colors))


# Your sign language to speech logic here
def sign_language_to_speech(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


# 1. New detection variables
sequence = []
previous_letter = None  # Store the previous detected letter
word = [""]  # Store the detected word
letter_detected = ""  # Store the most recently detected letter
threshold = 0.8
hand_detected = False  # Flag to track hand detection

combineWord = ""
sentence = ""
previous_word = ""
word1 = ""

add_to_sentence = False
speak_sentence = False
back_space = False

autocorrect_dict = {}


# Create a function to handle button click
def button_click(event, x, y, flags, param):
    global back_space  # Backspace Button
    if event == cv2.EVENT_LBUTTONDOWN:
        if 310 <= x <= 520 and 455 <= y <= 485:
            back_space = True

    global add_to_sentence  # OK Button
    if event == cv2.EVENT_LBUTTONDOWN:
        if 570 <= x <= 660 and 455 <= y <= 485:
            add_to_sentence = True

    global speak_sentence  # Sentence Button
    if event == cv2.EVENT_LBUTTONDOWN:
        if 270 <= x <= 400 and 550 <= y <= 580:
            speak_sentence = True


def hover_handler(event, x, y, flags, param):
    # Backspace Button
    if 310 <= x <= 520 and 455 <= y <= 485:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(frame, (310, 455), (520, 485), (245, 117, 16), -1)
            cv2.putText(frame, "<- Backspace", (598, 479), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                        cv2.LINE_AA)
            back_space = True
        else:
            cv2.rectangle(frame, (310, 455), (520, 485), (145, 117, 16), -1)
            cv2.putText(frame, "<- Backspace", (598, 479), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                        cv2.LINE_AA)

    # OK Button
    if 570 <= x <= 660 and 455 <= y <= 485:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(frame, (570, 455), (660, 485), (245, 117, 16), -1)
            cv2.putText(frame, "OK", (598, 479), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            add_to_sentence = True
        else:
            cv2.rectangle(frame, (570, 455), (660, 485), (145, 117, 16), -1)
            cv2.putText(frame, "OK", (598, 479), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # SUBMIT Button
    if 270 <= x <= 400 and 550 <= y <= 580:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(frame, (270, 550), (400, 580), (245, 117, 16), -1)
            cv2.putText(frame, "SUBMIT", (290, 573), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            speak_sentence = True
        else:
            cv2.rectangle(frame, (270, 550), (400, 580), (145, 117, 16), -1)
            cv2.putText(frame, "SUBMIT", (290, 573), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


cap = cv2.VideoCapture(0)

window_width = 800
window_height = 600

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (280, 350), (255, 255, 255), 2)
        image, results = mediapipe_detection(cropframe, hands)

        # Check if hand is detected
        if results is not None and results.multi_hand_landmarks:
            hand_detected = True
        else:
            hand_detected = False

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                detected_letter = actions[np.argmax(res)]

                # 3. Viz logic
                if np.unique(res.argmax()) == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        # Check if the detected letter is different from the previous one
                        if detected_letter != previous_letter:
                            # If it's different, add it to the letter_detected and word
                            letter_detected = detected_letter
                            if hand_detected:
                                word.append(detected_letter)

                            # Update the previous letter
                            previous_letter = detected_letter


        except Exception as e:
            # Handle exceptions as needed
            newWord = word[-1]
            if newWord not in combineWord[-1:]:
                combineWord += newWord


        if back_space:
            try:
                # Remove the last character from combineWord if it is not empty
                if len(combineWord) > 0:
                    combineWord = combineWord[:-1]  # Remove the last character from combineWord
                    print("Backspace clicked. New combineWord:", combineWord)

                    # Remove the last character from the sentence as well
                if len(sentence) > 0:
                    sentence = sentence[:-1]  # Remove the last character from sentence
                    print("Backspace clicked. New sentence:", sentence)

                # Adjust the word list for the removed character
                if len(word[-1]) > 0:
                    word[-1] = word[-1][:-1]  # Remove the last character from the last word in the list
                else:
                    # If the last word in the list is empty, remove the empty entry
                    if len(word) > 1:
                        word.pop()

            except Exception as e:
                print("Backspace error:", e)

            back_space = False  # Reset the backspace flag

        # OK Button Logic
        if add_to_sentence:
            if combineWord != "":
                if sentence:
                    sentence += "" + combineWord
                else:
                    sentence = combineWord
                word = [""]
                combineWord = ""
            add_to_sentence = False
            sentence += " "

        enlarge_frame = cv2.resize(frame, (window_width, window_height))

        cv2.rectangle(enlarge_frame, (0, 0), (350, 48), (245, 117, 16), -1)
        cv2.putText(enlarge_frame, "Output: - " + letter_detected, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(enlarge_frame, "Word: - " + combineWord, (3, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(enlarge_frame, "Sentence: - " + sentence, (3, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2, cv2.LINE_AA)

        cv2.rectangle(enlarge_frame, (310, 455), (520, 485), (245, 117, 16), -1)
        cv2.putText(enlarge_frame, "<- Backspace", (320, 479), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        button_click(0, 0, 0, 0, 0)

        cv2.rectangle(enlarge_frame, (570, 455), (660, 485), (245, 117, 16), -1)
        cv2.putText(enlarge_frame, "OK", (598, 479), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        button_click(0, 0, 0, 0, 0)

        cv2.rectangle(enlarge_frame, (270, 550), (400, 580), (245, 117, 16), -1)
        cv2.putText(enlarge_frame, "SUBMIT", (290, 573), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        button_click(0, 0, 0, 0, 0)

        # button_click(0, 0, 0, 0, 0)

        # Show to screen
        cv2.imshow('OpenCV Feed', enlarge_frame)
        # Set the mouse callback functions
        cv2.setMouseCallback("OpenCV Feed", button_click)
        hover_handler(0, 0, 0, 0, 0)

        # SUBMIT Button Logic
        if speak_sentence:
            engine.say(sentence)
            engine.runAndWait()
            speak_sentence = False
            sentence = ""

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()