from dotenv import load_dotenv
from langchain_community.document_loaders import(
    TextLoader
)
import os
import tempfile

load_dotenv()
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


if __name__ == "__main__":
    load_text_file()
