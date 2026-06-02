from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import(
    TextLoader,
    WebBaseLoader,
    PyPDFLoader
)
from langchain_core.documents import Document
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


#使用Document对象
def doc_structure():
    doc=Document(
        page_content="hello",
        metadata={"source": r"https://example.com",
                  "author":"zhang",
                  "created_at":"2026.6.2",
                  "tag":"test"}
        
    )
    print(f"doc_content{doc.page_content}")
    print(f"  metadata: {doc.metadata}")

    #更新,是创造个新的替换原来那个
    #doc.page_content=doc.page_content+"agent"（这个做法是错的）
    #print(f"doc_content new{doc.page_content}")

    new_doc=Document(
        page_content=doc.page_content+" agent",
        metadata={**doc.metadata,"update":True}
    )
    print(f"new doc content {new_doc.page_content}")
    print(f"new doc metadata {new_doc.metadata}")
'''
输出这个
doc_contenthello
  metadata: {'source': 'https://example.com', 'author': 'zhang', 'created_at': '2026.6.2', 'tag': 'test'}
new doc content hello agent
new doc metadata {'source': 'https://example.com', 'author': 'zhang', 'created_at': '2026.6.2', 'tag': 'test', 'update': True}
'''

#使用pdf加载器解析pdf pypdfloader
def pdf_demo(pathstr):
    pdfloader=PyPDFLoader(pathstr)
    documents=pdfloader.load()
    for i,doc in enumerate(documents):
        print(f"{i+1}page content: {doc.page_content[:100]}")
        print(f"page{i+1} metadata: {doc.metadata}")
    '''
    输出这个
    1page content: LangChain Document Loaders - Demo Document
Understanding LangChain Document Loaders
1. Introduction

page1 metadata: {'producer': 'PyPDF', 'creator': 'PyPDF', 'creationdate': '2026-02-02T21:45:08+00:00', 'source': './docs/langchain_demo.pdf', 'total_pages': 3, 'page': 0, 'page_label': '1'}
2page content: LangChain Document Loaders - Demo Document
4. Best Practices
When working with document loaders, con
page2 metadata: {'producer': 'PyPDF', 'creator': 'PyPDF', 'creationdate': '2026-02-02T21:45:08+00:00', 'source': './docs/langchain_demo.pdf', 'total_pages': 3, 'page': 1, 'page_label': '2'}
3page content: LangChain Document Loaders - Demo Document
formatting, and provides realistic content for testing yo
page3 metadata: {'producer': 'PyPDF', 'creator': 'PyPDF', 'creationdate': '2026-02-02T21:45:08+00:00', 'source': './docs/langchain_demo.pdf', 'total_pages': 3, 'page': 2, 'page_label': '3'}
    '''


if __name__ == "__main__":
    #load_text_file()
    #web_loader()
    #lazy_loader()
    #doc_structure()
    pdf_demo("./docs/langchain_demo.pdf")
