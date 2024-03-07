import streamlit as st
import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_verbose
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains.question_answering import load_qa_chain
from prompts import RAG_PROMPT, WEB_SEARCH_PROMPT

set_verbose(True)

### Data Analysis with Python and Statistics RAG

# Cached data file
@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    """
    This function reads the data from the uploaded
    file either in CSV or Excel format.

    Returns
        A Pandas dataframe.
    """
    # extract file name
    filename = uploaded_file.name
    
    # check the file extension and read data
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a .csv, .xls, or .xlsx file.")
        return None

# Cached embeddings
@st.cache_resource
def get_docembeddings():
    """
    This function retrieves FAISS embeddings of books on Statistics stored locally.

    Returns
        docembeddings : Indexed embeddings 
    """
    # retrieve pre-saved embeddings
    docembeddings = FAISS.load_local("statistics_llm_faiss_index", OpenAIEmbeddings())
    return docembeddings
    

def get_answer_from_rag(query, qa_chain):
    """
    This function retrieves answer from a Q&A chain for 
    the question submitted by user.

    Returns
        output(dict): A dictionary with keys 'Answer' and 'Reference'

    """
    # perform similiarity search using user question
    relevant_chunks = docembeddings.similarity_search_with_score(query,k=5)

    chunk_docs=[]
    for chunk in relevant_chunks:
        chunk_docs.append(chunk[0])

    # invoke Q&A agent and format output
    results = qa_chain.invoke({
        "input_documents": chunk_docs, "question": query
        })

    text_reference = "\n".join(doc.page_content for doc in results["input_documents"])

    output={"Answer":results["output_text"],"Reference":text_reference}

    return output

def process_q_n_a(query, qa_chain):
    """
    This function handles triggering the Q&A agent and returning the 
    answer response. 

    Returns
        output(dict): A dictionary with keys 'Answer' and 'Reference'
    """
    try:
        if query:
            output = get_answer_from_rag(query, qa_chain)
            if not output or 'Answer' not in output or output['Answer'] is None:
                return {"Answer": "No answer available", "Reference": ""}
        else:
            return {"Answer": "Waiting for user to input query and hit Enter.", "Reference": ""}
    except Exception as e:
        print(e)
        return {"Answer": "Status: Failure --- some error occurred", "Reference": ""}
    return output

def save_conversation_and_display(response, option=None):
    """
    This function saves 'assistant' responses into session state
    variables for memory purposes.
    """
    # update state 'conversation_memory' and 'conversation_history' 
    st.session_state.conversation_memory.chat_memory.add_ai_message(response)
    st.session_state.conversation_history.append({"role": "assistant", "content": response})

    # write response to web app
    if option == 'Retrieval' or option == 'Web Search':
        st.chat_message("assistant").write(response)


### DuckDuckGo search

# Define search tool
web_search_tool = DuckDuckGoSearchRun()
tools = [web_search_tool]


### Streamlit Display and Functionality

st.title(":female-technologist: Stassi : A Statistics Assistant :bar_chart:")

# File uploader to get data from user
uploaded_file = st.file_uploader("Please upload data file for analysis.", type=['csv', 'xls', 'xlsx'])

# Retrieve the dataframe
df = None
if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write(df.head())

# Sidebar input for OpenAI API key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Initialize or clear the conversation memory
if "conversation_memory" not in st.session_state or st.sidebar.button("Clear Conversation History"):
    st.session_state.conversation_memory = ConversationBufferMemory(memory_key="chat_history")


# Initialize conversation history in session state
if "conversation_history" not in st.session_state or st.sidebar.button("Clear Conversation History"):
    st.session_state.conversation_history = []

# Display conversation history
for message in st.session_state.conversation_history:
    st.chat_message(message["role"]).write(message["content"])

# Display selectbox for various analysis pathways
option = st.sidebar.selectbox(
    'Analysis Pathway',
    ('None', 'Data Analysis', 'Retrieval', 'Web Search'))     

            
### Main functionality
    
if __name__ == "__main__":
    
    
    # User input handling
    if (input := st.chat_input(placeholder="How may I help you?")) and input.strip():
            
        if not openai_api_key:
            st.info("Please provide an OpenAI API key to continue.")
            st.stop()
            
        llm_gpt4 = ChatOpenAI(
                temperature=0, model="gpt-4", openai_api_key=openai_api_key, streaming=True
        )
            
        # Add user message to memory buffer and session state for display
        st.session_state.conversation_memory.chat_memory.add_user_message(input)
        st.session_state.conversation_history.append({"role": "user", "content": input})
        st.chat_message("user").write(input)
            
        if option == 'Data Analysis':
                
            if df is not None:
                conversation_context = st.session_state.conversation_memory.load_memory_variables({})
                   
                agent_pandas_df = create_pandas_dataframe_agent(
                    llm_gpt4,
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    handle_parsing_errors=True,
                )

                with st.chat_message("assistant"):
                    stmlt_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                        
                    response = agent_pandas_df.run({"input":conversation_context}, callbacks=[stmlt_callback])
                        
                    # Update both the conversation memory and session state for display
                    save_conversation_and_display(response)
            else:
                st.error("Data frame did not load. Please upload a valid data file.")
            
        elif option == 'Retrieval':
                
            # Define Q&A chain
            qa_chain = load_qa_chain(llm_gpt4, chain_type="map_rerank", return_intermediate_steps=True, prompt=RAG_PROMPT)
                            
            # Load and cache embedding generation for efficient retrieval
            docembeddings = get_docembeddings()

            # Process query
            response = process_q_n_a(input, qa_chain)

            # Update both the conversation memory and session state for display
            save_conversation_and_display(response['Answer'], option)
            
        elif option == 'Web Search':

            # Define a web search agent using DuckDuckGo search
            web_search_agent = create_react_agent(llm_gpt4, tools, WEB_SEARCH_PROMPT)
            agent_executor = AgentExecutor(agent=web_search_agent, tools=tools)
                
            # Execute web search to answer question
            response = agent_executor.invoke({"input": input})
             
            # Update both the conversation memory and session state for display
            save_conversation_and_display(response['output'], option)
        
        else:
            # If no analysis pathway chosen
            st.write("Please choose an analysis pathway.")
            
    
