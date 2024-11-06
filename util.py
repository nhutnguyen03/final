#Dòng này nhập thư viện numpy và gán tên rút gọn là np. Thư viện này giúp xử lý các phép tính toán số học, đặc biệt là các phép toán mảng, rất hữu ích trong xử lý hình ảnh và toán học về hình học.
import numpy as np

#Hàm này tính góc giữa ba điểm a, b, và c trong không gian 2D. Các điểm a, b, c đại diện cho các tọa độ (x, y) trên mặt phẳng.
def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

#Hàm này tính khoảng cách giữa hai điểm đầu tiên trong danh sách landmark_ist.
def get_distance(landmark_ist):
    if len(landmark_ist) < 2:
        return
    (x1, y1), (x2, y2) = landmark_ist[0], landmark_ist[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])