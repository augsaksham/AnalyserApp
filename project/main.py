from unittest import result
from fastapi import FastAPI
import predict as pre
app = FastAPI()

@app.get("/")
async def root():
    return {"greeting":"Hello world"}


@app.post('/some/{text},{reason}')
def predict(text:str,reason:str):
    preds=pre.predict(text,reason)
    result=str(preds[0])
    if preds[0]==1:
        result+=" (YES)"
    else :
        result+=" (NO)"
    return {"Input Reason": reason,"Predicted Reason":preds[1],"Result":result} 

