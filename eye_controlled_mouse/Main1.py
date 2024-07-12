import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam and FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

def detect_blink(eye_landmarks, frame_h, threshold=0.004):
    """Detect blink based on eye landmarks."""
    y1 = eye_landmarks[0].y * frame_h
    y2 = eye_landmarks[1].y * frame_h
    blink_detected = abs(y1 - y2) < threshold
    return blink_detected

while True:
    # Read a frame from the webcam
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Process the frame with FaceMesh
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # Highlight specific landmarks for tracking
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            if id == 1:
                # Adjust sensitivity by scaling the cursor movement
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)
        
        # Landmarks around the left eye
        left_eye_landmarks = [landmarks[145], landmarks[159]]
        for landmark in left_eye_landmarks:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # Landmarks around the right eye
        right_eye_landmarks = [landmarks[374], landmarks[386]]
        for landmark in right_eye_landmarks:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
        
        # Check for left blink
        left_blink = detect_blink(left_eye_landmarks, frame_h)
        if left_blink:
            print("Left blink detected")
            pyautogui.click(button='left')
            pyautogui.sleep(1)  # Pause to prevent multiple clicks
        
        # Check for right blink
        right_blink = detect_blink(right_eye_landmarks, frame_h)
        if right_blink:
            print("Right blink detected")
            pyautogui.click(button='right')
            pyautogui.sleep(1)  # Pause to prevent multiple clicks
    
    # Display the frame
    cv2.imshow('Eye Controlled Mouse', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV window
cam.release()
cv2.destroyAllWindows()
