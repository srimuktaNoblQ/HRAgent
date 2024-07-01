import os
import autogen
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

config_list = [{"model": "gpt-3.5-turbo-0125", 'api-key': os.getenv("OPENAI_API_KEY")}]

llm = OpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
)

embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
)



Settings.llm = llm
Settings.embed_model = embed_model

finance_agent = ReActAgent.from_tools(
    tools=[],  
    llm=llm,
    max_iterations=10,
    verbose=True,
    description="This agent handles finance-related queries."
)

hr_agent = ReActAgent.from_tools(
    tools=[],  
    llm=llm,
    max_iterations=10,
    verbose=True,
    description="This agent handles HR-related queries."
)

tech_agent = ReActAgent.from_tools(
    tools=[],  
    llm=llm,
    max_iterations=10,
    verbose=True,
    description="This agent handles tech-related queries."
)

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    human_input_mode="ALWAYS",
    code_execution_config=False,
)

groupchat = autogen.GroupChat(
    agents=[finance_agent, hr_agent, tech_agent, user_proxy],
    messages=[],
    max_round=500,
    speaker_selection_method="round_robin",
    enable_clear_history=True,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

chat_result = user_proxy.initiate_chat(
    manager,
    message="Can you explain the latest quarterly financial results?"
)

print(chat_result)
