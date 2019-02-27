import datetime

import face_recognition
from PIL import Image

from util import get_image_array, create_directory


class FaceCapture:
    def __init__(self, model="cnn"):
        self._model = model

    def _get_face_locations(self, image_array):
        return face_recognition.api.face_locations(image_array, model=self._model)

    def _save_faces(self, image_array, face_locations):
        date = str(datetime.datetime.now())
        num_face = len(face_locations)
        if num_face != 0:
            print("[INFO] [" + date + "] " + str(num_face) + " FACE DETECTED")
        count = 0
        date += " | "
        for face_loc in face_locations:
            count += 1
            try:
                face_array = image_array[face_loc[0] - 15:face_loc[2] + 15, face_loc[3] - 15:face_loc[1] + 15, :]
                img = Image.fromarray(face_array)
            except ValueError:
                face_array = image_array[face_loc[0]:face_loc[2], face_loc[3]:face_loc[1], :]
                img = Image.fromarray(face_array)
            try:
                img.save("saved_faces/" + date + str(count) + ".jpg")
            except FileNotFoundError:
                create_directory("saved_faces")
                img.save("saved_faces/" + date + str(count) + ".jpg")

    def capture_all_faces(self, image, from_file=False):
        if from_file:
            image = get_image_array(image)
        face_locations = face_recognition.api.face_locations(image)
        self._save_faces(image, face_locations)


if __name__ == "__main__":
    fc = FaceCapture()
    fc.capture_all_faces("crowd_face.jpg", from_file=True)
