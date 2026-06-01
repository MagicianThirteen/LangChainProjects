from dotenv import load_dotenv
from langchain_community.document_loaders import(
    TextLoader,
    WebBaseLoader
)
import os
import tempfile
from bs4 import BeautifulSoup

load_dotenv()

#使用txtloader
def load_text_file():
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt"
    )as temp_file:
        temp_file.write(
            b"hello"
        )
    temp_file_path=temp_file.name

    #使用txt加载器
    try:
        #这里的参数用文件路径
        txtloader=TextLoader(temp_file_path)
        documents=txtloader.load()

        for doc in documents:
            print("文档内容：")
            print(doc)
            print(doc.page_content)
        #输出：
        #文档内容：
        #page_content='hello' metadata={'source': 'C:\\Users\\MAGICI~1\\AppData\\Local\\Temp\\tmpfgsh411f.txt'}
        #hello
        
    finally:
        os.remove(temp_file_path)
    
#使用网页加载器
def web_loader():
    loader=WebBaseLoader(
        "https://www.baidu.com/"
    )
    documents=loader.load()

    print(f"Loaded{len(documents)} document(s) from baidu")
    print(f"source:{documents[0].metadata.get('source','N/A')}")
    print(f"Length:{documents[0].page_content} 字数")
    print(f"{documents[0].page_content[:10]}……")
    '''
    输出
        Loaded1 document(s) from baidu
    source:https://www.baidu.com/
    Length:






    字数







    ……
    '''


if __name__ == "__main__":
    #load_text_file()
    web_loader()
