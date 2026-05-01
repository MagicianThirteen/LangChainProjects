from dotenv import load_dotenv
from importlib.metadata import version

from core_concepts import langchain_core_concepts
load_dotenv()


from langchain_openai import ChatOpenAI
core_version=version("langchain_core")
lg_version=version("langgraph")

print(f"langchain_core version: {core_version}")
print(f"langgraph version: {lg_version}")




def main():
    print("Hello from langchainprojects!")
    llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
    response=llm.invoke("Say setup complete! in one word")
    print(f"Respose from ChatOpenAI:{response}")


if __name__ == "__main__":
    langchain_core_concepts()
    #main()
