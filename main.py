from fastapi import FastAPI, HTTPException 
from logger import logger 
from multi_agent import call_agent

app = FastAPI() 


@app.get("/") 
def home(): 
    return {"message": "FASTAPI is running..."}
    
@app.post("/chat") 
def chat(query): 
    try: 
        response = call_agent(query) 
        logger.info("Received chat request") 
        return {"response": response} 

    except Exception as e: 
        logger.error(f"Error : {e}") 
        raise HTTPException(status_code=500,detail=str(e))