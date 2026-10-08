from fastapi import FastAPI

app=FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Salam!"}

@app.get("/price")
def price():
    return {"product": "Urban X7", "price": "3,200 MAD"}