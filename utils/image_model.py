from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import os

# Load model once
model = MobileNetV2(weights='imagenet')

# Your Spoonacular API key
SPOONACULAR_API_KEY = "a12cd108e59a431fbff1cfc6a9d91345"

def predict_image(img_path):
    # Step 1: Predict label using MobileNetV2
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=1)[0][0]  # top-1 prediction
    label = decoded[1]
    confidence = float(decoded[2])

    # Step 2: Use Spoonacular API for nutrition + recipe
    try:
        # Search food item
        search_url = f"https://api.spoonacular.com/food/ingredients/search"
        search_params = {
            "query": label,
            "apiKey": SPOONACULAR_API_KEY
        }
        search_resp = requests.get(search_url, params=search_params)
        search_data = search_resp.json()
        item_id = search_data.get("results", [{}])[0].get("id")

        if not item_id:
            raise Exception("No Spoonacular food match found")

        # Get nutrition info
        info_url = f"https://api.spoonacular.com/food/ingredients/{item_id}/information"
        info_resp = requests.get(info_url, params={
            "amount": 100,
            "unit": "g",
            "apiKey": SPOONACULAR_API_KEY
        })
        info_data = info_resp.json()
        nutrients = {n["name"]: n["amount"] for n in info_data.get("nutrition", {}).get("nutrients", [])}

        protein = f"{nutrients.get('Protein', 'N/A')}g"
        fat = f"{nutrients.get('Fat', 'N/A')}g"
        carbs = f"{nutrients.get('Carbohydrates', 'N/A')}g"

        # Get 3 recipes
        recipe_url = "https://api.spoonacular.com/recipes/complexSearch"
        recipe_resp = requests.get(recipe_url, params={
            "query": label,
            "number": 3,
            "apiKey": SPOONACULAR_API_KEY
        })
        recipe_data = recipe_resp.json()
        recipes = [r["title"] for r in recipe_data.get("results", [])]

        return {
            "label": label,
            "confidence": confidence,
            "protein": protein,
            "fat": fat,
            "carbs": carbs,
            "recipes": {
                "breakfast": recipes[0] if len(recipes) > 0 else "Not found",
                "lunch": recipes[1] if len(recipes) > 1 else "Not found",
                "dinner": recipes[2] if len(recipes) > 2 else "Not found"
            }
        }

    except Exception as e:
        return {
            "label": label,
            "confidence": confidence,
            "protein": "Error",
            "fat": "Error",
            "carbs": "Error",
            "recipes": {
                "breakfast": "Error",
                "lunch": "Error",
                "dinner": "Error"
            },
            "error": str(e)
        }
