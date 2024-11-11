from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

template = """You are an assistant that answers questions based on the provided document.

DOCUMENT CONTENT:
{context}

RULES:
- Provide clear and concise answers based on the provided document content
- Focus on the relevant information from the document
- If the question cannot be answered from the document, respond with:
  "I apologize, but I could not find information related to your question in the document."

Current question: {question}"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

chat_prompt_template = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])