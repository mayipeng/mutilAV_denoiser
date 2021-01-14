import numpy as np
import cv2


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
MOUTH_INDICES = [i for i in range(48,68)]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def extract_mouth_center(shape):
    points = map(lambda i: shape.part(i), MOUTH_INDICES)
    points = list(points)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 20, sum(ys) // 20

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def rotated_landmarks(M, landmarks):
    i = 0;
    for part in landmarks.parts():
        landmarks.part(i).x = int(M[0][0]*part.x+M[0][1]*part.y+M[0][2])
        landmarks.part(i).y = int(M[1][0]*part.x+M[1][1]*part.y+M[1][2])
        i = i+1
    return landmarks

def crop_image(image, landmarks_np):
    bottom = np.max(landmarks_np[:,1])
    top = np.min(landmarks_np[:,1])
    left = np.min(landmarks_np[:,0])
    right = np.max(landmarks_np[:,0])
    return image[top:bottom, left:right]

def crop_image2(image, height, width, cx, cy, threshold=5):
    if cy - height < 0:                                                
        cy = height                                                    
    if cy - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if cx - width < 0:                                                 
        cx = width                                                     
    if cx - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if cy + height > image.shape[0]:                                     
        cy = image.shape[0] - height                                     
    if cy + height > image.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if cx + width > image.shape[1]:                                      
        cx = image.shape[1] - width                                      
    if cx + width > image.shape[1] + threshold:                          
        raise Exception('too much bias in width') 
    cropped = np.copy(image[ int(round(cy) - round(height)): int(round(cy) + round(height)),
                         int(round(cx) - round(width)): int(round(cx) + round(width))])
    return cropped


def landmarks_to_np(shape, dtype='int'):
    # 创建68*2用于存放坐标
    shape_np = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 遍历每一个关键点
    # 得到坐标
    for i in range(0, shape.num_parts):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)

    return shape_np