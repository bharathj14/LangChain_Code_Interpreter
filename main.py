from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_ollama import OllamaLLM


load_dotenv()


def main():
    print("start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code. 
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer. 
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    # Using OpenAi LLM -  gpt-4.1 or gpt-4o - paid APIs
    # agent = create_react_agent(
    #     prompt = prompt,
    #     llm=ChatOpenAI(
    #         temperature=0,
    #         model="gpt-4.1",  # or "gpt-4o"
    #     ),
    #     tools=tools,
    # )

    # Using OLlama LLM - deepSeek-coder, deepseek-r1, qwen2.5-coder:7b
    agent = create_react_agent(
        prompt = prompt,
        llm=OllamaLLM(
            model="qwen2.5-coder:7b",
            temperature=0,
        ),
        tools=tools,
    )

    agent_executor = AgentExecutor(agent=agent,tools=tools, verbose=True)

    agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 15 QRcodes that point to www.udemy.com/course/Langchain, you have qrcode package installed  already"""
        }
    )

if __name__ == "__main__":
    main()