import cv2
import numpy as np


def bboxes_intersect(box1, box2):
    """
    Check if two bounding boxes intersect
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Coordenadas dos cantos
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    return not (x1_max < x2 or x2_max < x1 or y1_max < y2 or y2_max < y1)


def intersects_on(bbox, bbox_list):
    """
    Return the index of the first bounding box that intersects with the given bbox
    """
    for i in range(len(bbox_list)):
        if bboxes_intersect(bbox, bbox_list[i]):
            return i
    return False


def merge_boxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calcula as novas coordenadas e dimensões
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y

    return (x, y, w, h)


def segmentation_boxes(image: np.ndarray) -> tuple:
    """
    Segment the image into bounding boxes
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 100:
            box_id = intersects_on((x, y, w, h), boxes)
            if box_id is not False:
                boxes[box_id] = merge_boxes(boxes[box_id], (x, y, w, h))
            else:
                boxes.append((x, y, w, h))
            # roi = image_gray[y:y+h, x:x+w]
    return boxes


def draw_boxes(
    image: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: tuple,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image
    """

    img = image.copy()
    # Configurações de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Tamanho do texto
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Fundo do texto
    cv2.rectangle(img, (x, y - h - 6), (x + w, y), color, -1)

    # Texto
    cv2.putText(
        img,
        label.upper(),
        (x, y - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return img
