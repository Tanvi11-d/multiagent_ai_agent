import os 
from logger import logger 
from dotenv import load_dotenv 
from fastapi import HTTPException 
from composio import Composio 
from composio_langchain import LangchainProvider 
from langchain_groq import ChatGroq 
from langchain.agents import create_agent 

load_dotenv() 

USER_ID = os.getenv("user_id") 
api_key=os.getenv("GROQ_API_KEY") 

composio = Composio(
    provider=LangchainProvider()
) 

llm = ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",
    api_key=api_key
) 

tools = composio.tools.get( 
    user_id=USER_ID, 
    tools=[
        "GMAIL_FETCH_EMAILS",
        "GMAIL_SEND_EMAIL",
        "GMAIL_FORWARD_MESSAGE", 
        "GMAIL_GET_DRAFT",
        "GMAIL_SEND_DRAFT"
    ] 
) 

system_prompt = """ When calling any tool:
- Keep responses concise and accurate.
- If field type is : 
    - string -> use " " (empty string, NEVER null). 
    - array -> use []. 
    - boolean -> use True/False.
    - integer -> use 0. 
- always follow the tool schema. 
- Fill required fields properly. 
- If user say fetch messages then your are show body or subject and from. """ 
 
agent = create_agent( 
    model=llm, 
    tools=tools, 
    system_prompt=system_prompt) 

def run_agent(query: list): 
    try: 
        result = agent.invoke({ "messages": query }) 
        reply = result["messages"][-1].content 
        logger.info("Agent response generated") 
        return reply 
    except Exception as e: 
        logger.error(f"Agent execution error: {e}") 
        raise HTTPException(status_code=500, detail=str(e))