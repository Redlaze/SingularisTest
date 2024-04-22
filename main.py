import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# объявление констант
CONFIDENCE_THRESHOLD = 0.2
VIOLET = (255, 0, 255)
WHITE = (255, 255, 255)


def start_tracking_detection():
    """
    Захват и анализ видео
    """
    while True:

        # время начала расчета fps
        start = datetime.datetime.now()
        ret, frame = video_cap.read()

        # если больше нет кадров для обработки, выход из цикла
        if not ret:
            break
        arr_obj = object_recognition(frame)
        tracker_lst = object_tracking(frame, arr_obj)

        # время окончания вычисления fps
        end = datetime.datetime.now()

        # показывает время, затраченное на обработку 1 кадра
        total = (end - start).total_seconds()
        print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
        result = create_track(tracker_lst, frame)

        # просмотр и запись кадров
        cv2.imshow("Frame", cv2.resize(result, (1920 // 2, 1080 // 2)))
        writer.write(result)
        if cv2.waitKey(1) == ord("q"):
            break

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()


def object_recognition(frame):
    """
    Распознавание и фильтрация объектов
    :param frame: кадр видео
    :return results: массив координат контуров распознанных объектов
    """

    # запуск модели YOLO
    # повышение яркости кадра для улучшения прогноза
    frame_detection = increase_brightness(frame, value=95)
    detections = model(frame_detection)[0]
    results = []
    for data in detections.boxes.data.tolist():

        # извлечение вероятности, связанной с обнаружением
        confidence = data[4]
        Id = int(data[5])

        # фильтрация объектов
        if classes[Id] in classes_to_look_for:
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            results.append([(xmin, ymin, xmax - xmin, ymax - ymin), confidence, class_id])
    return results


def object_tracking(frame_shot, array_detection):
    """
    трекинг объектов
    :param frame_shot: кадр видео
    :param array_detection: массив координат контуров распознанных объектов
    :return: массив координат объектов для маски
    """

    tracks = tracker.update_tracks(array_detection, frame=frame_shot)
    inc_track = 0
    tracker_list = []
    for track in tracks:
        if not track.is_confirmed() or inc_track > len(array_detection) - 1:
            continue

        # получение идентификатора трека и ограничивающей рамки
        track_id = track.track_id
        xmin, ymin, xmax, ymax = (int(array_detection[inc_track][0][0]), int(array_detection[inc_track][0][1]),
                                  int(array_detection[inc_track][0][2]) + int(array_detection[inc_track][0][0]),
                                  int(array_detection[inc_track][0][3]) + int(array_detection[inc_track][0][1]))
        class_indx = int(array_detection[inc_track][2])
        score = round(array_detection[inc_track][1], 2)

        # создание ограничивающей рамки и запись идентификатора объекта, класса, точность прогнозирования
        cv2.rectangle(frame_shot, (xmin, ymin),
                      (xmax,
                       ymax), VIOLET, 10)
        cv2.rectangle(frame_shot, (xmin, ymin - 40),
                      (xmin + 250, ymin), VIOLET, -1)
        font_size = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = 3

        text = f'Id:{track_id} {classes[class_indx]} {score}'
        cv2.putText(frame_shot, text, (xmin, ymin - 10),
                    font, font_size, WHITE, width, cv2.LINE_AA)
        delta = 40
        delta_rect = 250
        tracker_list.append([[xmin, ymin],
                             [xmin, ymin - delta],
                             [xmin + delta_rect, ymin - delta],
                             [xmin + delta_rect, ymin],
                             [xmax, ymin],
                             [xmax, ymax],
                             # [xmin + delta_rect, ymin],
                             [xmin, ymax],

                             ])
        inc_track += 1
    return tracker_list


def create_track(tracker_lst, frame):
    """
    вырезание объектов и создание маски
    :param tracker_lst: массив координат объектов для маски
    :param frame_shot: кадр видео
    :return: вырезанные объекты из кадра
    """

    contours = [np.array(i) for i in tracker_lst]
    # создание маски
    stencil = np.zeros(frame.shape, dtype='uint8')
    for i in tracker_lst:
        stencil[i[0][1]:i[5][1], i[0][0]: max(i[4][0], i[2][0])] = [255, 255, 255]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result = cv2.bitwise_and(frame, stencil)
    return result


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


if __name__ == '__main__':
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    # Определение классов, которым будет присвоен приоритет при поиске по изображению
    # Имена указаны в файле coco.names.txt
    video = input("Введите путь файла (URL): ")
    look_for = input("Метка класса: ").split(',')
    tracker = DeepSort(max_age=5000)
    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for

    # инициализация объекта видеозахвата
    video_cap = cv2.VideoCapture(video)

    # инициализация объекта для записи видео
    writer = cv2.VideoWriter("result.avi",
                             cv2.VideoWriter_fourcc(*"MJPG"), 20, (1920, 1080))

    # загрузка предварительно обученной модели YOLOv8m
    model = YOLO("Resources/yolov8m.pt")

    start_tracking_detection()
