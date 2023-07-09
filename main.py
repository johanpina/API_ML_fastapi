from  fastapi import FastAPI
import pickle


# Cargamos el modelo previamente entrenado
with open('models/modelRF.pkl', 'rb') as Gb:
    modelo = pickle.load(Gb)


app = FastAPI()

@app.get('/')
def hello():
    return {'message': 'Hello World'}

@app.post('/predict')
def predict(request:dict):
    # Get the data from the POST request.
    data = request['data']
    # Make prediction using model loaded from disk as per the data.
    prediction = modelo.predict(data)
    # Take the first value of prediction
    output = prediction[0]

    return {'prediction': output}