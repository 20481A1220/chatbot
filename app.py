from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
from urllib.parse import quote
import os

# Initialize the database connection
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    encoded_password = quote(password)
    db_uri = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# SQL generation chain setup
def get_sql_chain(db):
    template = """
    You are a god-level SQL expert with extensive knowledge of PostgreSQL. You are a data analyst at a company. 
    You are interacting with a user who is asking you questions about the company's database. Based on the table schema below, 
    write a PostgreSQL SQL query that would answer the user's question. 

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the PostgreSQL SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    Your turn:
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Query execution and response generation
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, SQL query, and SQL response, write a natural language response. 
    You should not give the query as the output and show the number rather than text.

    If the data does not exist or the question cannot be answered, respond with: 
    "I don't have the necessary data to answer this query."

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: safe_query_execution(db, vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# Safely execute the SQL query and catch any errors
def safe_query_execution(db, query):
    try:
        return db.run(
            query.replace("DATE_SUB(CURRENT_DATE", "CURRENT_DATE - INTERVAL")
                 .replace("joiningdate", "joiningdate::timestamp")
        )
    except Exception as e:
        return f"An error occurred while running the query: {str(e)}"

# Main Streamlit app setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat with PostgreSQL", page_icon=":speech_balloon:")

st.title("Chat with PostgreSQL")

# Sidebar for database connection settings
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using PostgreSQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="5432", key="Port")
    st.text_input("User", value="postgres", key="User")
    st.text_input("Password", type="password", value="2003", key="Password")
    st.text_input("Database", value="emp", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("Connected to database!")
            except Exception as e:
                st.error(f"Failed to connect to the database: {str(e)}")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Get user input
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    if "db" in st.session_state:
        with st.chat_message("AI"):
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
    else:
        st.error("Please connect to the database first.")
