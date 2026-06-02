from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import(
    TextLoader,
    WebBaseLoader,
)
from langchain_community.document_loaders import DirectoryLoader
print("加载Directoryloader的包")
import os
import tempfile
from pathlib import Path
from bs4 import BeautifulSoup



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

#加载器加载目录里面对应的格式文件
def lazy_loader():
    #建立临时目录，临时目录写几个文件
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            path=Path(tmpdir)/f"doc_{i}.txt"
            path.write_text(f"hello{i}")
        #创建DirectoryLoader,注意几个参数
        #因为with会自动删除，所以这些操纵要写在with里面
        loader=DirectoryLoader(
            tmpdir,
            glob="*.txt",
            loader_cls=TextLoader,
            use_multithreading=True
        )
        #然后打印对应的pagecontent和metadata
        for doc in loader.load():
            print(f"doc:{doc.page_content}")
            print(f"doc metadata:{doc.metadata['source'][:10]}")
'''
输出这个
加载Directoryloader的包
doc:hello0
doc metadata:C:\Users\M
doc:hello2
doc metadata:C:\Users\M
doc:hello1
doc metadata:C:\Users\M
'''


if __name__ == "__main__":
    #load_text_file()
    #web_loader()
    lazy_loader()
