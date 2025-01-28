from flask import Flask, request, jsonify
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
from flask_cors import CORS


app = Flask(__name__)

# Ajouter CORS
CORS(app)


# Charge le modèle ONNX
session = ort.InferenceSession("fashion_mnist.onnx")

def preprocess_image(image):
    """Prétraiter l'image avant de l'envoyer au modèle"""
    # Convertir l'image en grayscale et redimensionner
    image = image.convert("L")  # Convertir en noir et blanc
    image = image.resize((28, 28))  # Redimensionner à 28x28 (taille de l'entrée du modèle)
    image = np.array(image) / 255.0  # Normalisation des pixels entre 0 et 1
    # Appliquer la normalisation avec une moyenne de 0.5 et un écart-type de 0.5
    image = (image - 0.5) / 0.5  # Normalisation comme dans le modèle d'entraînement
    image = image.reshape(1, 1, 28, 28).astype(np.float32)  # Adapter la forme pour le modèle
    print("Image prétraitée :", image.shape)
    return image



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Vérifie si un fichier est envoyé
        if 'file' not in request.files:
            print("Aucun fichier envoyé.")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        print(f"Fichier reçu : {file.filename}")

        if file.filename == '':
            print("Fichier sans nom.")
            return jsonify({'error': 'No file selected'}), 400

        # Ouvrir et prétraiter l'image
        image = Image.open(file)
        preprocessed_image = preprocess_image(image)  # Utilisation de la fonction de prétraitement
        print("Image prétraitée pour le modèle.")

        # Charger le modèle ONNX et effectuer la prédiction
        input_name = session.get_inputs()[0].name
        print("Nom de l'entrée du modèle :", input_name)

        prediction = session.run(None, {input_name: preprocessed_image})[0]
        print("Résultat brut de la prédiction :", prediction)

        # Identifier la classe prédite
        class_idx = int(np.argmax(prediction))
        classes = [
            "T-shirt/top", "Pantalon", "Pull-over", "Robe",
            "Manteau", "Sandale", "Chemise", "Basket",
            "Sac", "Bottine"
        ]
        predicted_label = classes[class_idx]

        print("Classe prédite :", predicted_label)

        return jsonify({'prediction': predicted_label})
    
    except Exception as e:
        print("Erreur :", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
