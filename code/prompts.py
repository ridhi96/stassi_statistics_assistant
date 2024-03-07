from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser

### Prompt template for Statistics RAG

prompt_rag ="""
## Goal
- To accurately answer a statistical and mathematical question using provided context and conversation history. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

## Constraints
- Always include answer and score in output.
- Exclude any non-mathematical or nonsensical text present in the context, highlighting only the contextual 
explanations and interpretations of mathematical content. 
- While answering, maintain the logical and conceptual integrity of the mathematical information in the context.
- Do not attempt to directly interpret or retain the actual mathematical symbols in the output.
- Do not create your own interpretations of mathematical symbols or expressions or equations if they are not 
extracted correctly in the context or are incomplete or incorrect.

## Skill
- Expertise in using the context to answer questions about a range of statistical techniques and their practical 
applications.
- Ability to differentiate between meaningful mathematical content and irrelevant or gibberish information in the 
context.

## Workflow
Use the following pieces of context to answer the question at the end using the context and conversation.

Output should be in the following format:

Question: [question here]
Answer: [answer here]
Score: [score between 0 and 100]

Begin!

Context:
---------
{context}
---------
Question: {question}
Answer:"""

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

Begin!

Context:
---------
{context}
---------
Question: {question}
"""

RAG_PROMPT = PromptTemplate(
    template=prompt_rag,
    input_variables=["context", "question"],
    output_parser=output_parser
)

### Prompt template for DuckDuckGo search

search_template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

WEB_SEARCH_PROMPT = PromptTemplate.from_template(search_template)