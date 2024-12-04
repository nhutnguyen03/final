import cv2
import mediapipe as mp
import pyautogui
import util
from pynput.mouse import Button, Controller
import time
import math

mouse = Controller()

# Screen size and smoothening factor
screen_width, screen_height = pyautogui.size()
smoothening = 1

# Initialize previous mouse position and tracking variables
prev_x, prev_y = 0, 0
move_accuracy_count = 0
move_attempts = 0
click_accuracy_count = 0
click_attempts = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None

def move_mouse(index_finger_tip):
    global prev_x, prev_y, move_accuracy_count, move_attempts
    if index_finger_tip is not None:
        # Map the fingertip's normalized coordinates to the full screen dimensions
        target_x = int(index_finger_tip.x * screen_width)
        target_y = int(index_finger_tip.y * screen_height)

        # Apply smoothing
        smooth_x = prev_x + (target_x - prev_x) / smoothening
        smooth_y = prev_y + (target_y - prev_y) / smoothening

        # Update previous positions for next frame
        prev_x, prev_y = smooth_x, smooth_y

        # Calculate movement accuracy
        move_attempts += 1
        if math.hypot(target_x - smooth_x, target_y - smooth_y) < 10:  # Threshold for accurate move
            move_accuracy_count += 1

        # Move the mouse
        pyautogui.moveTo(int(smooth_x), int(smooth_y))

def is_left_click(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
        thumb_index_dist > 50
    )

def is_right_click(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
        thumb_index_dist > 50
    )

def is_double_click(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist > 50
    )

def detect_gesture(frame, landmark_list, processed):
    global click_accuracy_count, click_attempts
    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        if util.get_distance([landmark_list[4], landmark_list[5]]) < 50 and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            click_attempts += 1
            click_accuracy_count += 1  # Assuming each detected click is a correct click
        elif is_left_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            click_attempts += 1
            click_accuracy_count += 1
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            click_attempts += 1
            click_accuracy_count += 1

def calculate_accuracy():
    move_accuracy = (move_accuracy_count / move_attempts) * 100 if move_attempts > 0 else 0
    click_accuracy = (click_accuracy_count / click_attempts) * 100 if click_attempts > 0 else 0
    
    # Print and save to file
    accuracy_text = f"Move Accuracy: {move_accuracy:.2f}%\nClick Accuracy: {click_accuracy:.2f}%\n"
    print(accuracy_text)
    with open("accuracy_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{accuracy_text}\n")

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # Biến tính FPS
    prev_time = time.time()

    try:
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

            # Tính toán FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Hiển thị FPS trên khung hình
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Tay thay chuot', frame)

            # Log accuracy results every 10 seconds
            if time.time() - start_time > 10:
                calculate_accuracy()
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        calculate_accuracy()  # Final accuracy log on exit

if __name__ == '__main__':
    main()

