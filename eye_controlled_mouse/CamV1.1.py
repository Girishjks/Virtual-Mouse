import cv2
import mediapipe as mp
import pyautogui
import time
import webbrowser
import random

# Initialize camera and mediapipe face mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Initialize blink tracking variables
blink_count = 0
last_blink_time = 0
text_display_time = 0
DISPLAY_DURATION = 5  # Duration to display each text message in seconds
TEXT_DELAY = 2  # Delay between consecutive text messages
DOUBLE_BLINK_TIME = 0.5  # Maximum time interval for detecting a double blink
TRIPLE_BLINK_TIME = 1.5  # Time window to detect triple blinks

# Random messages to display
random_messages = [
    "This is Just an example?",
    "How texts can be configured!",
    "Next we will integrate Morse Code",
    "Thanks and Go Sleep",
    "Signing off as Your man Girish!",
]

# Initialize text display parameters
current_message_index = 0
text_to_display = random_messages[current_message_index]
next_display_time = 0
text_display_index = 0
text_display_timer = 0
CHAR_DISPLAY_SPEED = 0.05  # Time delay between displaying each character

def display_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 255, 255), thickness=2):
    global text_display_index, text_display_timer, CHAR_DISPLAY_SPEED
    
    if text_display_index < len(text):
        if time.time() - text_display_timer > CHAR_DISPLAY_SPEED:
            text_display_timer = time.time()
            text_display_index += 1
    
    text_to_display = text[:text_display_index]
    cv2.putText(frame, text_to_display, position, font, font_scale, color, thickness)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # Detect blink
        if (left_eye[0].y - left_eye[1].y) < 0.004:
            current_time = time.time()
            if current_time - last_blink_time < DOUBLE_BLINK_TIME:
                blink_count += 1
            else:
                blink_count = 1
            last_blink_time = current_time

            if blink_count == 2:
                text_display_time = current_time
                # Display random message
                random_message = random.choice(random_messages)
                print(f"Displaying message: {random_message}")
                text_to_display = random_message
                text_display_index = 0  # Reset text display index
                text_display_timer = time.time()  # Reset text display timer
                blink_count = 0
        else:
            if time.time() - last_blink_time > TRIPLE_BLINK_TIME:
                blink_count = 0  # Reset blink count if time window is exceeded

    # Display text if double blink detected
    if time.time() - text_display_time < DISPLAY_DURATION:
        display_text(frame, text_to_display, (50, 50))
    elif time.time() > next_display_time:
        # Display next message if it's time
        current_message_index = (current_message_index + 1) % len(random_messages)
        text_to_display = random_messages[current_message_index]
        text_display_time = time.time()  # Update display time for new message
        next_display_time = text_display_time + DISPLAY_DURATION + TEXT_DELAY

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
