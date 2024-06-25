import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Función para extraer características LBP
def extract_lbp_features(image, radius, n_points, method):
    lbp = local_binary_pattern(image, n_points, radius, method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalización del histograma
    return hist

# Cargar el modelo SVM entrenado
def load_trained_svm_model():
    # Ruta del dataset
    dataset_path = r'C:\Users\orlan\OneDrive\Escritorio\ULTIMO SEMESTRE\Ciencia de datos\Proyecto\\'
    
    # Cargar los archivos CSV de entrenamiento y prueba
    train_df = pd.read_csv(os.path.join(dataset_path, 'lbp_train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_path, 'lbp_test.csv'))

    # Separar características y etiquetas
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Definir el modelo SVM
    svm_model = SVC(kernel='linear', random_state=42)

    # Definir los parámetros para GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    # Aplicar GridSearchCV para encontrar los mejores parámetros
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train_encoded)

    # Obtener el mejor modelo
    best_svm_model = grid_search.best_estimator_
    
    return best_svm_model, le

# Cargar el modelo entrenado
svm_model, le = load_trained_svm_model()

# Configuración de la interfaz gráfica
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento")
        self.root.geometry("800x600")
        
        self.label = tk.Label(root, text="Toma la foto para verificar si es real o falso")
        self.label.pack()
        
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        
        self.detect_button = tk.Button(self.button_frame, text="Detectar rostro", command=self.detect_face)
        self.detect_button.pack(side=tk.LEFT)
        
        self.capture_button = tk.Button(self.button_frame, text="Tomar foto", command=self.capture_photo)
        self.capture_button.pack(side=tk.LEFT)
        
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_face(self):
        self.cap = cv2.VideoCapture(0)
        self.update_frame()
        
    def update_frame(self):
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.root.after(10, self.update_frame)
        
    def capture_photo(self):
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            messagebox.showinfo("Resultado", "No se detecto nada")
        else:
            (x, y, w, h) = faces[0]  # Assuming the first detected face is the target
            face_roi = gray[y:y+h, x:x+w]
            lbp_features = extract_lbp_features(face_roi, radius=1, n_points=8, method='uniform')
            lbp_features = lbp_features.reshape(1, -1)
            prediction = svm_model.predict(lbp_features)
            label = le.inverse_transform(prediction)[0]
            
            messagebox.showinfo("Resultado", f"Deteccion: {label}")
        
        self.cap.release()
        self.cap = None
        
# Crear la aplicación
root = tk.Tk()
app = FaceRecognitionApp(root)
root.mainloop()
