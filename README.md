# stassi_statistics_assistant

Stassi is a Streamlit app that is designed to assist in solving statistical data analysis questions in an easy-to-use chatbot format courtesy of GPT-4 by OpenAI.

A user can input a columnar data set (up to 200 MB) and analyze it by asking questions based on statistical methods. 

The app provides three pathways for analysis:
  1) Data Analysis – The user input data is read into a Pandas dataframe which can be interacted with by asking questions about the data. In the back-end Python code is executed to provide answers.
  2) Retrieval – A Retrieval Augmented Generation(RAG) based LLM can access data from some books on Statistics to answer questions of a technical nature.
  3) Web Search – An option is provided to query the internet in case additional information is required or user wants to check more sources.

Technologies used: LangChain, Streamlit

## Repository Structure

### code

'data_analysis_llm.py'
- Contains bulk of the implementation of RAG, Python based data analysis, and web search.

'prompts.py'
- Contains the prompts used by LangChain agents and chains.

'RAG_embeddings.py'
- Contains code to generate and locally store FAISS embeddings of some books on Statistics.

### docs

Directory for books used by the RAG. Book sources listed below:

* [Econometric Analysis of Cross Section and Panel Data Jeffrey M. Wooldridge](https://ipcig.org/evaluation/apoio/Wooldridge%20-%20Cross-section%20and%20Panel%20Data.pdf)
* [Econometric Data Science: A Predictive Modeling Approach by Francis X. Diebold](https://www.sas.upenn.edu/~fdiebold/Teaching104/Econometrics.pdf)
* [Probability and Statistics: The Science of Uncertainty by Michael J. Evans and Jeffrey S. Rosenthal](https://www.utstat.toronto.edu/mikevans/jeffrosenthal/book.pdf)

### statistics_faiss_llm_index

Contains the relevant FAISS embeddings for easy retrieval generated by executing RAG_embeddings.py.

## Running Stassi

To run the app follow the steps:

1) Clone the repo
2) Run the streamlit app by executing: streamlit run ~/code/data_analysis_llm.py
3) There is no need to re-generate the embeddings as they are already present in the repo. But if you would like to regenerate them, then execute following using command line: python3 ~/code/RAG_embeddings.py
4) Once the app is running in your browser, enter your OpenAI API key with access to GPT-4 in the sidebar widget to get started.
