# python 파일로 실행 가능하나, data 화면에서 깨짐이 발생할 수 있음으로 추천하지 않음
# 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
import cv2
import dlib
import tensorflow as tf
import pickle
import time
import re

# 상수 정의
SCREEN_WIDTH = 1280 # 화면 너비
SCREEN_HEIGHT = 720 # 화면 높이
BUTTON_WIDTH = 300 # 버튼 너비
BUTTON_HEIGHT = 100 # 버튼 높이
BUTTON_COLOR = (100, 100, 100) # 버튼 색상
BUTTON_COLOR_HOVER = (150, 150, 150) # 버튼 호버 색상
TEXT_COLOR = (255, 255, 255) # 텍스트 색상
FONT = cv2.FONT_HERSHEY_SIMPLEX # 폰트
FONT_SCALE = 1.5 # 버튼 폰트 크기
FONT_THICKNESS = 3 # 버튼 폰트 두께1
FONT_THICKNESS_SLIM = 2 # 버튼 폰트 두께2
LINE_TYPE = cv2.LINE_AA # 선 종류
WINDOW_NAME = "READING HELPER" # 창 이름

# 모델 로드, 기본 데이터, 파라미터 설정 (고민재)
detector = dlib.get_frontal_face_detector() # 얼굴 검출기
predictor = dlib.shape_predictor('shape_68.dat') # 얼굴 landmark 검출기

left = [36, 37, 38, 39, 40, 41] # 왼쪽 눈 landmark
right = [42, 43, 44, 45, 46, 47] # 오른쪽 눈 landmark
kernel = np.ones((9, 9), np.uint8) # 커널
threshold = 0 # 동공 검출 threshold

model = tf.keras.models.load_model('default_model.keras') # default eye-tracking 모델 (미리 학습시켜 놓음)

# default landmark 불러오기
landmarks = None
with open('default_landmarks.pkl', 'rb') as file:
    landmarks = pickle.load(file)


# default 보정 데이터 불러오기
default_eye_datas = None
default_screen_positions = None
with open('default_calibration_data.pkl', 'rb') as file:
    default_eye_datas, default_screen_positions = pickle.load(file)

# 영상 처리 변수
cap = None

# 보정 데이터 생성 과정 파라미터
horizontal_number = 3
vertical_number = 3

eye_datas = []
screen_positions = []

# 매끄러운 움직임 처리 파라미터
num = 3
std = 1.5
alpha = 0.4

# 텍스트 읽기 관련 함수 및 파라미터 (임지윤)
# 영어 텍스트 파일 읽기
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

# 두 문장씩 나누기
def split_text(text):
    # 문장 구분
    sent_end = re.compile(r'(?<=[.!?]) +')
    sent = sent_end.split(text)
    
    # 두 문장씩 묶기
    group_sent = []
    for i in range(0, len(sent), 2):
        if i+1 < len(sent):
            group_sent.append(sent[i] + " " + sent[i+1])
        else:
            group_sent.append(sent[i])
    if group_sent[-1] == '': group_sent = group_sent[:-1]

    return group_sent
    
# 텍스트 입력
def draw_centered_text(img, texts, max_line_length, n):
    global space, thick1, thick2
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 2
    thickness = thick1
    color = (0, 0, 0)
    
    y0, dy = 300, t_y
    width = img.shape[1]

    x, y = 0, y0
    
    space = []

    a=-1

    for i, line in enumerate(texts[n].split('\n')):
        for j, char in enumerate(line):
            a += 1
            if j % max_line_length == 0 and j != 0:
                y += dy
                x = 0

            x_offset = (width - max_line_length * t_x) // 2 + x * t_x
            y_offset = y
            
            cv2.putText(img, char, (x_offset, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)

            top_left = (x_offset, y_offset - t_y)
            bottom_right = (x_offset + t_x, y_offset)
            space.append([top_left, bottom_right])
            
            x += 1

        y += dy
        x = 0

    return space

# bionic reading 기능 구현
# 텍스트 입력 (bold)
def draw_centered_text_bold(img, texts, max_line_length, n):
    global space, thick1, thick2, t_x, t_y, FONT
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 2
    thickness = thick1
    thickness2 = thick2
    color = (0, 0, 0)
    
    y0, dy = 300, t_y
    width = img.shape[1]

    x, y = 0, y0
    
    space = []

    a = -1

    for i, line in enumerate(texts[n].split('\n')):
        words = line.split(' ')
        for word_index, word in enumerate(words):
            bold_len = len(word) // 2

            for j, char in enumerate(word):
                a += 1
                if x % max_line_length == 0 and x != 0:
                    y += dy
                    x = 0

                x_offset = (width - max_line_length * t_x) // 2 + x * t_x
                y_offset = y

                # 각 단어의 앞 n/2개의 알파벳을 볼드 처리
                if j < bold_len:
                    cv2.putText(img, char, (x_offset, y_offset), font, font_scale, color, thickness2, cv2.LINE_AA)
                else:
                    cv2.putText(img, char, (x_offset, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)

                top_left = (x_offset, y_offset - t_y)
                bottom_right = (x_offset + t_x, y_offset)
                space.append([top_left, bottom_right])
                
                x += 1

            if word_index < len(words) - 1:
                a += 1
                if x % max_line_length == 0 and x != 0:
                    y += dy
                    x = 0
                x_offset = (width - max_line_length * t_x) // 2 + x * t_x
                y_offset = y
                space.append([(x_offset, y_offset - t_y), (x_offset + t_x, y_offset)]) 
                x += 1 

        y += dy
        x = 0

    return space


# 시선이 몇 번째 문자 위에 있는지 반환
def onText(point, page):
    global min_x, min_y, max_x, max_y, max_line_length, texts
    if (min_x < point[0] < max_x) & (min_y < point[1] < max_y):
        if ((point[1] - min_y) // t_y) * max_line_length + ((point[0] - min_x) // t_x) < len(texts[page]):
            return int(((point[1] - min_y) // t_y) * max_line_length + ((point[0] - min_x) // t_x))
        else: return -1
    else: return -1

# 시선이 n번째 문자 근처인지 반환
def nearText(n, point):
    global space, plus
    if n >= len(space): return False
    elif space[n][0][0] - plus <= point[0] <= space[n][1][0] + plus:
        if space[n][0][1] - plus <= point[1] <= space[n][1][1] + plus:
            return True
    else: return False
    
# 텍스트 읽기 관련 파라미터
text_file_path = 'text.txt'
max_line_length = 22  # 한 줄에 표시할 최대 문자 수
text = read_text_file(text_file_path)
t_x, t_y = 50, 70 # 텍스트가 차지하는 x길이, y길이
len_t = len(text)
haveread = [0 for _ in range(len_t)]
min_x, max_x, min_y, max_y = 0, 0, 0, 0
space = []
plus = 200 # 텍스트 주변이라고 인식하는 범위
now = -1 # 현재 몇번째 텍스트 읽고 있는지
thick1 = 1 # 일반 텍스트 두께
thick2 = 5 # 볼드체 두께
texts = split_text(text) # 두 문장씩 나누어진 텍스트 리스트

# 시각적 피드백 및 데이터 처리 관련 함수 및 파라미터 (김도윤)
#두 점 사이 거리
def distance(pts1,pts2):
    return int(((pts1[0]-pts2[0])**2+(pts1[1]-pts2[1])**2)**0.5)

#원 그리기
radius=1
cir_list=[[] for _ in range(len(texts))]
color=(0,255,0)

def circle(point,lst,page):
    global radius
    found=False
    for _ in range(len(lst[page])):
        if point in lst[page][_]:
            lst[page][_][1]+=0.1            
            found=True
            break
        
    if not found:
        lst[page].append([point,radius])
    

def draw_circle(frame,lst,page):
    global color
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            cv2.circle(frame,lst[i][j][0],lst[i][j][1],color,-1)
draw=0
            
#time
def time_difference_in_hms(seconds1, seconds2):
    # 두 시간의 차이를 절대값으로 계산
    diff = abs(seconds1 - seconds2)
    
    # 시, 분, 초로 변환
    hours = diff // 3600
    minutes = (diff % 3600) // 60
    seconds = diff % 60
    
    return hours, minutes, seconds

times=[0,0]
sentence_times=[0 for _ in range(len(texts))]

def time_difference_in_sec(seconds1,seconds2):
    diff = abs(seconds1 - seconds2)
    return diff

use_time=[time_difference_in_sec(sentence_times[i],sentence_times[i+1]) for i in range(len(texts)-1)]

# 그래프 그리기
def draw_plot(list_x, list_y):
    fig, ax = plt.subplots()
    ax.plot(list_x, list_y, 'b-')
    ax.set_xlabel("page")
    ax.set_ylabel("time(Sec)")
    
    #축 최소 범위 설정
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) #정수만 축에 나타냄
    
    canvas = FigureCanvas(fig)
    fig.canvas.draw()
    
    # 캔버스를 이미지로 변환
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    
    plt.close() #그래프 닫기
    
    return img

# 기능 함수 정의 (고민재)
# 얼굴 landmark 검출기 객체를 ndarray로 변환
def shape_to_np(shape, dtype='int'):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# 사람이 한 명인지 확인
def is_one_person(img, rects):
    if len(rects) == 0:
        text = "No Person!"
        text_width, text_height = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = (img.shape[1] - text_width) // 2
        text_y = (img.shape[0] - text_height) // 2
        cv2.putText(img, text, (text_x, text_y), FONT, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
    elif len(rects) > 1:
        text = "Too Many People!"
        text_width, text_height = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = (img.shape[1] - text_width) // 2
        text_y = (img.shape[0] - text_height) // 2
        cv2.putText(img, text, (text_x, text_y), FONT, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
    else:
        return True
    return False

# 눈 위치 마스크 생성
def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

# 트랙바 nothing 콜백 함수
def nothing(x):
    pass

# 동공 검출
def contouring(thresh, mid, img, right=False, show_pos=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(cnt)
        if right:
            cx += mid
        if show_pos:
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        return cx, cy, area
    except:
        return None, None, None

# 모델 입력으로 사용되는 눈 데이터 계산
def calculate_eye_data(left_points, right_points, left_pupil, right_pupil):
    left_diff = left_points - left_pupil[:2]
    right_diff = right_points - right_pupil[:2]
    diff = np.hstack((left_diff, right_diff)).flatten()
    eye_data = np.hstack((diff, left_pupil[2:3], right_pupil[2:3]))
    return eye_data

# 눈 데이터로부터 화면 위치 추정
def estimate_screen_position(eye_data, num=1, std=3):
    eye_data = np.array(eye_data).reshape(1, -1)
    if num > 1:
        eye_data_batch = np.tile(eye_data, (num, 1))
        noise = np.random.normal(0, std, eye_data_batch.shape)
        eye_data_batch += noise
        screen_positions = model.predict(eye_data_batch)
        screen_position = np.mean(screen_positions, axis=0)
    else:
        screen_position = model.predict(eye_data).flatten()
    return screen_position

# 매끄러운 움직임 처리
def smooth_position(new_position, smoothed_position, alpha=0.5):
    if smoothed_position is None:
        return new_position
    else:
        return alpha * new_position + (1 - alpha) * smoothed_position

# 화면 구성
# 어느 화면에서나 q를 눌러 프로그램 종료
# calibraiton 화면에서는 c를 눌러 데이터 저장
# 즉, 버튼 클릭, q, c면 충분한 조작 가능
current_screen = "main_menu"
is_hovered = []

# 둥근 버튼 그리는 함수 (https://stackoverflow.com/questions/18973103/how-to-draw-a-rounded-rectangle-rectangle-with-rounded-corners-with-opencv)
def rounded_rectangle(src, top_left, bottom_right, radius=0.1, color=255, thickness=1, line_type=cv2.LINE_AA):
    p1 = top_left
    p3 = bottom_right
    p2 = (bottom_right[0], top_left[1])
    p4 = (top_left[0], bottom_right[1])

    width = abs(p2[0] - p1[0])
    height = abs(p4[1] - p1[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * min(width, height) / 2)

    if thickness < 0:
        cv2.rectangle(src, (p1[0] + corner_radius, p1[1]), (p3[0] - corner_radius, p3[1]), color, thickness)
        cv2.rectangle(src, (p1[0], p1[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, thickness)

        cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color, thickness, line_type)
        cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color, thickness, line_type)
        cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90, color, thickness, line_type)
        cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90, color, thickness, line_type)

    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p4[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color, thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color, thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90, color, thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90, color, thickness, line_type)

    return src

# 버튼 그리는 함수
def draw_button(image, label, position, color):
    x, y = position
    rounded_rectangle(image, (x, y), (x + BUTTON_WIDTH, y + BUTTON_HEIGHT), 0.6,color, -1)
    text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    text_x = x + (BUTTON_WIDTH - text_size[0]) // 2
    text_y = y + (BUTTON_HEIGHT + text_size[1]) // 2
    cv2.putText(image, label, (text_x, text_y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, LINE_TYPE)

# 버튼 클릭 여부 확인
def button_click(position, button_position):
    x, y = position
    bx, by = button_position
    return bx <= x <= bx + BUTTON_WIDTH and by <= y <= by + BUTTON_HEIGHT

# 버튼 호버 여부 확튼
def button_hover(position, button_position):
    x, y = position
    bx, by = button_position
    return bx <= x <= bx + BUTTON_WIDTH and by <= y <= by + BUTTON_HEIGHT

# 메인 메뉴 (보정, 텍스트 읽기, 종료)
# 이하 다른 메뉴들도 구성은 동일
def main_menu(screen):
    global current_screen, is_hovered
    # 버튼 목록
    buttons = [
        {"label": "CALIBRATION", "position": (92, 532), "action": "calibration"},
        {"label": "READ TEXT", "position": (492, 532), "action": "read_text_menu"},
        {"label": "EXIT", "position": (892, 532), "action": "exit_program"}
    ]
    is_hovered = [False] * len(buttons)

    # 마우스 콜백 함수 (버튼 클릭 구현)
    def mouse_callback(event, x, y, flags, param):
        global current_screen
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                current_screen = button["action"]
                return

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    main_menu_img=cv2.imread("UI-main_menu.png", cv2.IMREAD_COLOR)
    main_menu_img_resize=cv2.resize(main_menu_img,(SCREEN_WIDTH,SCREEN_HEIGHT))
    # 메인 메뉴 화면 구현
    while current_screen == "main_menu":
        screen[:] = 0

        # 메인 메뉴 이미지 불러오기
        screen[0:main_menu_img_resize.shape[0], 0:main_menu_img_resize.shape[1], :] = main_menu_img_resize.copy()

        # 버튼 그리기
        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)

        cv2.imshow(WINDOW_NAME, screen)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            exit_program()

# 보정 메뉴 (landmark, 동공 검출 thresbold, 눈 데이터)
def calibration(screen):
    global current_screen, is_hovered
    buttons = [
        {"label": "LANDMARKS", "position": (172, 513), "action": "landmarks_calibration"},
        {"label": "THRESHOLD", "position": (492, 513), "action": "threshold_calibration"},
        {"label": "EYE DATAS", "position": (812, 513), "action": "eye_datas_calibration"},
        {"label": "BACK", "position": (20,20 ), "action": "main_menu"}
    ]
    is_hovered = [False] * len(buttons)

    def mouse_callback(event, x, y, flags, param):
        global current_screen
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                current_screen = button["action"]
                return

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    calibration_img=cv2.imread("UI-calibration.png",cv2.IMREAD_COLOR)
    calibration_img_resize=cv2.resize(calibration_img,((SCREEN_WIDTH,SCREEN_HEIGHT)))
    while current_screen == "calibration":
        screen[:] = 0

        screen[0:calibration_img_resize.shape[0], 0:calibration_img_resize.shape[1], :] = calibration_img_resize.copy()

        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)

        cv2.imshow(WINDOW_NAME, screen)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            exit_program()

# 텍스트 읽기 메뉴 (테스트, 결과, 뒤로)
def read_text_menu(screen):
    global current_screen, is_hovered
    buttons = [
        {"label": "TEST", "position": (250, 513), "action": "test"},
        {"label": "DATA", "position": (700,513), "action": "data"},
        {"label": "BACK", "position": (20, 20), "action": "main_menu"}
    ]
    is_hovered = [False] * len(buttons)

    def mouse_callback(event, x, y, flags, param):
        global current_screen
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                if button["action"] == "Test": haveread = [0 for i in range(len_t)]
                current_screen = button["action"]
                return

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    read_text_img=cv2.imread("UI-read_text.png",cv2.IMREAD_COLOR)  
    read_text_img_resize=cv2.resize(read_text_img,(SCREEN_WIDTH,SCREEN_HEIGHT))
    while current_screen == "read_text_menu":
        screen[:] = 0

        screen[0:read_text_img_resize.shape[0], 0:read_text_img_resize.shape[1], :] = read_text_img_resize.copy()

        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)

        cv2.imshow(WINDOW_NAME, screen)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            exit_program()

# 텍스트 읽기 테스트 (뒤로, ON)
def test(screen):
    global current_screen, is_hovered, num, std, alpha, threshold, cap, left, right, kernel, landmarks, min_x, min_y, max_x, max_y, now, haveread
    buttons = [
        {"label": "ON", "position": (875, 100), "action": "bold_test"},
        {"label": "BACK", "position": (100, 100), "action": "read_text_menu"}
    ]
    is_hovered = [False] * len(buttons)

    def mouse_callback(event, x, y, flags, param):
        global current_screen, cap
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                if cap is not None:
                    cap.release()
                    cap = None
                current_screen = button["action"]
                return

    cap = cv2.VideoCapture(0)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    estimated_screen_pos = None
    smoothed_screen_pos = None

    page = 0

    test_img=cv2.imread("UI-test.png",cv2.IMREAD_COLOR)  
    test_img_resize=cv2.resize(test_img,(1280,720))
    while current_screen == "test":
        screen[:] = 0
        screen[0:test_img_resize.shape[0], 0:test_img_resize.shape[1], :] = test_img_resize.copy()

        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)

        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        
        estimated_screen_pos = None

        # 텍스트 입력
        space = draw_centered_text(screen, texts, max_line_length, page)
        
        # 텍스트 영역 범위 변수 저장
        min_x, max_x = space[0][0][0], space[min(len(texts[page]),max_line_length)-1][1][0]
        min_y, max_y = space[0][0][1], space[len(texts[page])-1][1][1]

        global times, sentence_times, draw

        # 읽기 시작한 시각과 끝난 시각 저장
        if page==0 and now==0:
            time_s=time.time()
            times[0]=int(time_s)
            
        elif page==len(texts)-1 and now==len(texts[page])-1:
            time_e=time.time()
            times[1]=int(time_e)
           
        
        # 각 페이지당 읽은 시간 저장
        time_stamp=time.time()
        sentence_times[page]=int(time_stamp)

        
        # 사람이 한 사람일때만 실행
        if is_one_person(img,rects):
            # 눈 데이터 계산 과정
            rect = rects[0]    
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = eye_on_mask(mask, left, shape)
            mask = eye_on_mask(mask, right, shape)
            mask = cv2.dilate(mask, kernel, 5)
            
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)
            thresh = cv2.medianBlur(thresh, 3)
            thresh = cv2.bitwise_not(thresh)

            left_pupil = contouring(thresh[:, 0:mid], mid, img, show_pos=True)
            right_pupil = contouring(thresh[:, mid:], mid, img, True, show_pos=True)
            
            left_points = shape[left].copy().astype(np.float32)
            right_points = shape[right].copy().astype(np.float32)
            
            # 눈 데이터를 바탕으로 화면 위치 추정
            if left_pupil[0] is not None and right_pupil[0] is not None:
                eye_data = calculate_eye_data(left_points, right_points, left_pupil, right_pupil)
                estimated_screen_pos = estimate_screen_position(eye_data, num, std)
                smoothed_screen_pos = smooth_position(estimated_screen_pos, smoothed_screen_pos, alpha)

            
        if smoothed_screen_pos is not None:

            # 시선 위치 표시
            cv2.circle(screen, (int(smoothed_screen_pos[0]*SCREEN_WIDTH), int(smoothed_screen_pos[1]*SCREEN_HEIGHT)), 30, (0, 255, 0), -1)
            current_pos = (int(smoothed_screen_pos[0] * SCREEN_WIDTH), int(smoothed_screen_pos[1] * SCREEN_HEIGHT))
            word_pos=((space[now][0][0]+space[now][1][0])//2,(space[now][0][1]+space[now][1][1])//2)

            # 실시간 피드백 문구 띄우기
            if distance(current_pos,word_pos) > 200:
                current_pos = np.array(current_pos)
                current_pos -= np.array([180,-10])
                cv2.putText(screen, "Return to text", current_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            # 읽고 있는 텍스트 표시
            if onText((smoothed_screen_pos[0]*SCREEN_WIDTH,smoothed_screen_pos[1]*SCREEN_HEIGHT),page) != -1:
                cv2.rectangle(screen, space[onText((smoothed_screen_pos[0]*SCREEN_WIDTH,smoothed_screen_pos[1]*SCREEN_HEIGHT),page)][0], space[onText((smoothed_screen_pos[0]*SCREEN_WIDTH,smoothed_screen_pos[1]*SCREEN_HEIGHT),page)][1], (0,255,255), -1)

            # 지금까지 읽은 텍스트 저장
            if nearText(now+1, (smoothed_screen_pos[0]*SCREEN_WIDTH,smoothed_screen_pos[1]*SCREEN_HEIGHT)):
                now += 1
                haveread[now] = 1
            
            # 읽은 텍스트 하이라이트
            for i in range(len_t):
                if haveread[i] > 0:
                    cv2.rectangle(screen, space[i][0], space[i][1], (0,255,255), -1)
            draw_centered_text(screen, texts, max_line_length, page)


        cv2.imshow(WINDOW_NAME, screen)
        key = cv2.waitKey(1) & 0xFF

        # 1 누르면 전 페이지, 2 누르면 다음 페이지
        if key == ord('1'):
            if page == 0: pass
            else: 
                page -= 1
                now = -1
                haveread = [0 for i in range(len_t)]

        if key == ord('2'):
            if page == len(texts) - 1: pass
            else: 
                page += 1
                now = -1
                haveread = [0 for i in range(len_t)]

        if key == ord('q'):
            exit_program()
    
    if cap is not None:
        cap.release()
        cap = None

# 볼드 텍스트 읽기 테스트 (뒤로)
def bold_test(screen):
    global current_screen, is_hovered, num, std, alpha, threshold, cap, left, right, kernel, landmarks, min_x, min_y, max_x, max_y, now, haveread
    buttons = [
        {"label": "OFF", "position": (875, 100), "action": "test"},
        {"label": "BACK", "position": (100, 100), "action": "read_text_menu"}
    ]
    is_hovered = [False] * len(buttons)

    def mouse_callback(event, x, y, flags, param):
        global current_screen, cap
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                if cap is not None:
                    cap.release()
                    cap = None
                current_screen = button["action"]
                return

    cap = cv2.VideoCapture(0)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    estimated_screen_pos = None
    smoothed_screen_pos = None

    page = 0

    btest_img=cv2.imread("UI-test.png",cv2.IMREAD_COLOR)  
    btest_img_resize=cv2.resize(btest_img,(1280,720))
    while current_screen == "bold_test":
        screen[:] = 0
        screen[0:btest_img_resize.shape[0], 0:btest_img_resize.shape[1], :] = btest_img_resize.copy()

        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)

        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        
        estimated_screen_pos = None

        #텍스트 입력
        space = draw_centered_text_bold(screen, texts, max_line_length, page)

        # 텍스트 영역 범위 변수 저장
        min_x, max_x = space[0][0][0], space[min(len(texts[page]),max_line_length)-1][1][0]
        min_y, max_y = space[0][0][1], space[len(texts[page])-1][1][1]     

        global times, sentence_times, draw

        # 읽기 시작한 시각과 끝난 시각 저장
        if page==0 and now==0:
            time_s=time.time()
            times[0]=int(time_s)
            
        elif page==len(texts)-1 and now==len(texts[page])-1:
            time_e=time.time()
            times[1]=int(time_e)
        
        # 각 페이지당 읽은 시간 저장
        time_stamp=time.time()
        sentence_times[page]=int(time_stamp)

        
        # 사람이 한 사람일때만 실행
        if is_one_person(img,rects):
            # 눈 데이터 계산 과정
            rect = rects[0]    
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = eye_on_mask(mask, left, shape)
            mask = eye_on_mask(mask, right, shape)
            mask = cv2.dilate(mask, kernel, 5)
            
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)
            thresh = cv2.medianBlur(thresh, 3)
            thresh = cv2.bitwise_not(thresh)

            left_pupil = contouring(thresh[:, 0:mid], mid, img, show_pos=True)
            right_pupil = contouring(thresh[:, mid:], mid, img, True, show_pos=True)
            
            left_points = shape[left].copy().astype(np.float32)
            right_points = shape[right].copy().astype(np.float32)
            
            # 눈 데이터를 바탕으로 화면 위치 추정
            if left_pupil[0] is not None and right_pupil[0] is not None:
                eye_data = calculate_eye_data(left_points, right_points, left_pupil, right_pupil)
                estimated_screen_pos = estimate_screen_position(eye_data, num, std)
                smoothed_screen_pos = smooth_position(estimated_screen_pos, smoothed_screen_pos, alpha)                

        if smoothed_screen_pos is not None:


            current_pos = (int(smoothed_screen_pos[0] * SCREEN_WIDTH), int(smoothed_screen_pos[1] * SCREEN_HEIGHT))
            word_pos=((space[now][0][0]+space[now][1][0])//2,(space[now][0][1]+space[now][1][1])//2)

            # 시선 위치 표시
            cv2.circle(screen, (int(smoothed_screen_pos[0]*SCREEN_WIDTH), int(smoothed_screen_pos[1]*SCREEN_HEIGHT)), 30, (0, 255, 0), -1)
            
            # 실시간 피드백 문구 띄우기
            if distance(current_pos,word_pos) > 200:
                current_pos = np.array(current_pos)
                current_pos -= np.array([180,-10])
                cv2.putText(screen, "Return to text", current_pos, FONT, 1.5, (0, 0, 255), 2)


            # 보고 있는 텍스트 표시
            if onText((smoothed_screen_pos[0]*SCREEN_WIDTH,smoothed_screen_pos[1]*SCREEN_HEIGHT),page) != -1:
                cv2.rectangle(screen, space[onText((smoothed_screen_pos[0]*SCREEN_WIDTH,smoothed_screen_pos[1]*SCREEN_HEIGHT),page)][0], space[onText((smoothed_screen_pos[0]*SCREEN_WIDTH,smoothed_screen_pos[1]*SCREEN_HEIGHT),page)][1], (0,255,255), -1)

            # 마지막으로 읽은 텍스트 저장
            if nearText(now+1, (smoothed_screen_pos[0]*SCREEN_WIDTH,smoothed_screen_pos[1]*SCREEN_HEIGHT)):
                now += 1
                haveread[now] = 1
            
            # 지금까지 읽은 텍스트 표시
            for i in range(len_t):
                if haveread[i] > 0:
                    cv2.rectangle(screen, space[i][0], space[i][1], (0,255,255), -1)
            draw_centered_text_bold(screen, texts, max_line_length, page)


        cv2.imshow(WINDOW_NAME, screen)
        key = cv2.waitKey(1) & 0xFF

        # 1 누르면 전 페이지, 2 누르면 다음 페이지
        if key == ord('1'):
            if page == 0: pass
            else: 
                page -= 1
                now = -1
                haveread = [0 for i in range(len_t)]

        if key == ord('2'):
            if page == len(texts) - 1: pass
            else: 
                page += 1
                now = -1
                haveread = [0 for i in range(len_t)]

        if key == ord('q'):
            exit_program()
    
    if cap is not None: 
        cap.release()
        cap = None

# 데이터 화면 (뒤로)
def data(screen):
    global current_screen, is_hovered
    buttons = [
        {"label": "BACK", "position": (92, 530), "action": "read_text_menu"}
    ]
    is_hovered = [False] * len(buttons)

    def mouse_callback(event, x, y, flags, param):
        global current_screen
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                current_screen = button["action"]
                return

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    data_img=cv2.imread("UI-data.png",cv2.IMREAD_COLOR)
    data_img_resize=cv2.resize(data_img,(SCREEN_WIDTH,SCREEN_HEIGHT))
    while current_screen == "data":
        screen[:] = 0
        screen[0:data_img_resize.shape[0], 0:data_img_resize.shape[1], :] = data_img_resize.copy()

        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)

        key = cv2.waitKey(1) & 0xFF
        # 시간 관련 계산
        global times,sentence_times,texts
        h,m,s=time_difference_in_hms(times[0],times[1])
        if len(texts)==1:
            use_time=[sentence_times[0]-times[0]]
        else:
            use_time=[sentence_times[0]-times[0]]+[time_difference_in_sec(sentence_times[i],sentence_times[i+1]) for i in range(len(texts)-1)]
        pages=[i+1 for i in range(len(texts))]
        
        # 화면에 그래프 표시
        plot_img=draw_plot(pages,use_time)
        x, y = 540,160
        screen[y:y+plot_img.shape[0], x:x+plot_img.shape[1], :] = plot_img.copy()
        cv2.putText(screen,f"TIME READ = {h :02.0f}:{m :02.0f}:{s :02.0f}",(490,180),FONT,1.5,(0,0,0),3)
        
        cv2.imshow(WINDOW_NAME, screen)
        
        if key == ord('q'):
            exit_program()

# landmakrs 보정 화면 (뒤로)   
def landmarks_calibration(screen):
    global current_screen, is_hovered, cap, landmarks
    buttons = [
        {"label": "Back", "position": (95, 150), "action": "calibration"}
    ]
    is_hovered = [False] * len(buttons)

    def mouse_callback(event, x, y, flags, param):
        global current_screen
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                current_screen = button["action"]
                return

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    cap = cv2.VideoCapture(0)
    alert_timer = 0

    landmarks_calibration_img=cv2.imread("UI-landmarks_calibration.png",cv2.IMREAD_COLOR)  
    landmarks_calibration_img_resize=cv2.resize(landmarks_calibration_img,(SCREEN_WIDTH,SCREEN_HEIGHT))
    while current_screen == "landmarks_calibration":
        screen[:] = 0
        screen[0:landmarks_calibration_img_resize.shape[0], 0:landmarks_calibration_img_resize.shape[1], :] = landmarks_calibration_img_resize.copy()

        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)

        # 얼굴 landmark 그리기
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if is_one_person(img, rects):
            rect = rects[0]    
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            
            for (x, y) in shape:
                cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        
        
        key = cv2.waitKey(1) & 0xFF

        x = 530
        y = 150
        screen[y:y+img.shape[0], x:x+img.shape[1]] = img
        
        # c를 누르면 현재 상태의 landmark 저장 (추후에 사용하기 위해)
        if key == ord('c') and is_one_person(img, rects):
            alert_timer = 20
            with open('landmarks.pkl', 'wb') as file:
                pickle.dump(shape, file)
            with open('landmarks.pkl', 'rb') as file:
                landmarks = pickle.load(file)            
        
        # 저장 알림을 일정 시간동안 표시
        if alert_timer > 0:
            cv2.putText(screen, "FACIAL LANDMARK SAVED!", (640, 620), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            alert_timer -= 1
            
        cv2.imshow(WINDOW_NAME, screen)
        
        if key == ord('q'):
            exit_program()
    if cap is not None:
        cap.release()
        cap = None

# 동공 검출 threshold 보정 화면 (뒤로)
def threshold_calibration(screen):
    global current_screen, is_hovered, cap, landmarks, kernel, threshold
    buttons = [
        {"label": "Back", "position": (95, 150), "action": "calibration"}
    ]
    is_hovered = [False] * len(buttons)

    def mouse_callback(event, x, y, flags, param):
        global current_screen
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                current_screen = button["action"]
                return

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    thresh = np.zeros_like(img)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    
    # 트랙바 사용이 어려워 별도의 화면을 사용하여 threshold 보정
    threshold_screen = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Threshold', thresh.shape[1], thresh.shape[0]+30)
    cv2.moveWindow('Threshold', 690, 200)
    cv2.createTrackbar('threshold', 'Threshold', 0, 255, nothing)
    threshold_saved = False

    threshold_calibration_img=cv2.imread("UI-threshold_calibration.png",cv2.IMREAD_COLOR)  
    threshold_calibration_img_resize=cv2.resize(threshold_calibration_img,(1280,720))
    while current_screen == "threshold_calibration":
        screen[:] = 0

        screen[0:threshold_calibration_img_resize.shape[0], 0:threshold_calibration_img_resize.shape[1], :] = threshold_calibration_img_resize.copy()

        threshold_screen[:] = 0
        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)

        # threshold가 저장되기 전
        if not threshold_saved:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = np.zeros_like(img)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for (x, y) in landmarks:
                cv2.circle(img, (x, y), 2, (200, 200, 200), -1)
            
            if is_one_person(img, rects):
                # 눈 데이터 계산 과정
                rect = rects[0]    
                shape = predictor(gray, rect)
                shape = shape_to_np(shape)
                
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask = eye_on_mask(mask, left, shape)
                mask = eye_on_mask(mask, right, shape)
                mask = cv2.dilate(mask, kernel, 5)
                
                eyes = cv2.bitwise_and(img, img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                threshold = cv2.getTrackbarPos('threshold', 'Threshold')
                
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=4)
                thresh = cv2.medianBlur(thresh, 3)
                thresh = cv2.bitwise_not(thresh)

                for (x, y) in shape:
                    cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
            
            # threshold에 따른 동공 검출 결과를 색으로 표시
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            thresh_green = np.zeros_like(thresh)
            thresh_green[:, :] = [0, 255, 0]
            img = cv2.bitwise_and(img, cv2.bitwise_not(thresh)) + cv2.bitwise_and(thresh_green, thresh)
            
            threshold_screen[0:img.shape[0], 0:img.shape[1], :] = img
        
        key = cv2.waitKey(1) & 0xFF
        
        # c를 누르면 threshold 저를
        if key == ord('c') and not threshold_saved:
            threshold_saved = True
            cap.release()
            cv2.destroyWindow('Threshold')
        
        # 저장되었을 때 화면에 표시
        if threshold_saved:
            cv2.putText(screen, "THRESHOLD SAVED!", (700, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        if not threshold_saved:
            cv2.imshow("Threshold", threshold_screen)
        
        cv2.imshow(WINDOW_NAME, screen)
        if key == ord('q'):
            exit_program()
    if cap is not None:
        cap.release()
    try:
        cv2.destroyWindow('Threshold')
    except:
        pass

# 눈 데이터 보정 화면 (보정, 뒤로)
def eye_datas_calibration(screen):
    global current_screen, is_hovered, cap, landmarks, kernel, threshold, horizontal_number, vertical_number, eye_datas, screen_positions
    buttons = [
        {"label": "TRAIN", "position": (95, 150), "action": "train"},
        {"label": "BACK", "position": (95, 300), "action": "calibration"}
    ]
    is_hovered = [False] * len(buttons)

    def mouse_callback(event, x, y, flags, param):
        global current_screen, cap
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                if cap is not None:
                    cap.release()
                    cap = None
                current_screen = button["action"]
                return

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    cap = cv2.VideoCapture(0)
    
    # 눈 데이터와 대응되는 화면의 점을 저장하여 모델 학습에 활용
    eye_datas = []
    screen_positions = []
    count = 0

    collecting = False

    eye_datas_calibration_img=cv2.imread("UI-eye_datas_calibration.png",cv2.IMREAD_COLOR)
    eye_datas_calibration_img_resize=cv2.resize(eye_datas_calibration_img,(1280,720))
    while current_screen == "eye_datas_calibration":
        screen[:] = 0

        screen[0:eye_datas_calibration_img_resize.shape[0], 0:eye_datas_calibration_img_resize.shape[1], :] = eye_datas_calibration_img_resize.copy()

        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)

        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (200, 200, 200), -1)
        
        if is_one_person(img, rects):
            # 눈 데이터 계산 과정
            rect = rects[0]    
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = eye_on_mask(mask, left, shape)
            mask = eye_on_mask(mask, right, shape)
            mask = cv2.dilate(mask, kernel, 5)
            
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)
            thresh = cv2.medianBlur(thresh, 3)
            thresh = cv2.bitwise_not(thresh)

            left_pupil = contouring(thresh[:, 0:mid], mid, img, show_pos=True)
            right_pupil = contouring(thresh[:, mid:], mid, img, True, show_pos=True)
            
            left_points = shape[left].copy().astype(np.float32)
            right_points = shape[right].copy().astype(np.float32)
            
            for (x, y) in shape:
                cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

        x = 530
        y = 130
            
        screen[y:y+img.shape[0], x:x+img.shape[1]] = img
        
        # 현재 보정에 사용하는 점을 화면에 표시
        screen_position = (count % horizontal_number / (horizontal_number-1), count // horizontal_number / (vertical_number-1))
        cv2.circle(screen, (int(screen_position[0] * SCREEN_WIDTH), int(screen_position[1] * SCREEN_HEIGHT)), 10, (0, 255, 0), -1)
        
        key = cv2.waitKey(1) & 0xFF
        # c를 누르면 현재 눈 데이터와 화면 위치 저장
        if key == ord('c'):
            if not collecting and left_pupil[0] is not None and right_pupil[0] is not None:
                collecting = True
                screen_positions.append((count % horizontal_number / (horizontal_number-1), count // horizontal_number / (vertical_number-1)))
                count += 1
                count %= horizontal_number * vertical_number
                left_pupil = np.array(left_pupil).astype(np.float32)
                right_pupil = np.array(right_pupil).astype(np.float32)
                eye_datas.append(calculate_eye_data(left_points, right_points, left_pupil, right_pupil))
        else:
            collecting = False 
                   
        cv2.imshow(WINDOW_NAME, screen)
        
        if key == ord('q'):
            exit_program()
    if cap is not None:
        cap.release()
        cap = None

# 눈 데이터로 모델 학습 (뒤로)
def train(screen):
    global current_screen, is_hovered, cap, landmarks, eye_datas, screen_positions
    buttons = [
        {"label": "BACK", "position": (92, 530), "action": "calibration"}
    ]
    is_hovered = [False] * len(buttons)

    def mouse_callback(event, x, y, flags, param):
        global current_screen
        for i, button in enumerate(buttons):
            is_hovered[i] = button_hover((x, y), button["position"])
            if event == cv2.EVENT_LBUTTONDOWN and button_click((x, y), button["position"]):
                current_screen = button["action"]
                return

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    screen[:] = 0
    cv2.imshow(WINDOW_NAME, screen)
    
    train_timer = 0
    train_finished = False
    train_img=cv2.imread("UI-train.png",cv2.IMREAD_COLOR)  
    train_img_resize=cv2.resize(train_img,(1280,720))
    while current_screen == "train":
        screen[:] = 0

        screen[0:train_img_resize.shape[0], 0:train_img_resize.shape[1], :] = train_img_resize.copy()

        for i, button in enumerate(buttons):
            button_color = BUTTON_COLOR_HOVER if is_hovered[i] else BUTTON_COLOR
            draw_button(screen, button["label"], button["position"], button_color)
        
        # 학습이 완료되었을 때 화면에 표시
        if train_finished:
            cv2.putText(screen, "TRAIN FINISHED", (500, 572), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 눈 데이터와 화면 위치로 모델을 학습 (상황, 개인에 따라 다른 모델을 사용해야 함)
        if not train_finished:
            train_timer += 1
            cv2.putText(screen, "TRAINING...", (500, 572), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if train_timer == 10:
                train_finished = True
                X = np.array(eye_datas)
                y = np.array(screen_positions)
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=200, batch_size=32)
            
        cv2.imshow(WINDOW_NAME, screen)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            exit_program()
    if cap is not None:
        cap.release()
        cap = None

# 프로그램 종료 (화면 종료, 카메라 해제)
def exit_program():
    global current_screen, cap
    current_screen = None
    cv2.destroyAllWindows()
    if cap is not None:
        cap.release()
        cap = None

# 화면 관리 함수 
if __name__ == "__main__":
    screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while current_screen:
        if current_screen == "main_menu":
            main_menu(screen)
        elif current_screen == "calibration":
            calibration(screen)
        elif current_screen == "read_text_menu":
            read_text_menu(screen)
        elif current_screen == "test":
            test(screen)
        elif current_screen == "bold_test":
            bold_test(screen)
        elif current_screen == "data":
            data(screen)
        elif current_screen == "landmarks_calibration":
            landmarks_calibration(screen)
        elif current_screen == "threshold_calibration":
            threshold_calibration(screen)
        elif current_screen == "eye_datas_calibration":
            eye_datas_calibration(screen)
        elif current_screen == "train":
            train(screen)
        elif current_screen == "exit_program":
            exit_program()