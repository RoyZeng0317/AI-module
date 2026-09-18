import face_recognition, cv2, pickle

known_face_list = [
    {
        "name": "Mary",
        "filename": "mary.jpg",
        "face_encoding": None
    },
    {
        "name": "Roy",
        "filename": "Roy.jpg",
        "face_encoding": None
    }
]