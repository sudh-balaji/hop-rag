from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

template = """You are a Security Officer. Provide direct answers based on this information:

INFORMATION:
{context}

RULES:
- Answer directly without mentioning documents
- Don't repeat the question
- Don't use chapter references
- Be clear and concise

Question: {question}

Answer:"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

chat_prompt_template = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])