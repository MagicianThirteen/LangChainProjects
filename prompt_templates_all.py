
from dotenv import load_dotenv
load_dotenv()
#常见通用模板
#基础模板，简单问答,把一段文字翻译成对应的语言{text}{language}
#输出结果：“宁瑶真棒”可以翻译为 “Ning Yao is really awesome” 或者 “Ning Yao is great.”
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
def basic_prompt_template():
    prompt=ChatPromptTemplate.from_template("把{text}翻译成{language}")
    message=prompt.format_messages(text="宁瑶真棒",language="英文")
    model=init_chat_model(model="gpt-4o-mini",temperature=0)
    response=model.invoke(message)
    print(response.content)
    












if __name__ == "__main__":
    basic_prompt_template()