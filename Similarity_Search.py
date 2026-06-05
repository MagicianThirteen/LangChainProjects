from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
import tempfile

load_dotenv()

def Similarity_Search():
    #准备文档,准备问题
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]
    query = "What programming languages exist?"
    #把文档和问题都向量化
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    doc_vects=embedding.embed_documents(docs)
    query_vect=embedding.embed_query(query)
    #余弦相似函数
    def cosine_similarity(vect1,vect2):
        return (np.dot(vect1,vect2))/(np.linalg.norm(vect1)*
                                      np.linalg.norm(vect2))
        
    #把文档向量和问题向量依次用余弦相似函数对比
    #这里要注意是计算好的向量之间在算相似度
    similarities=[cosine_similarity(query_vect,doc_vect) for doc_vect in doc_vects ]
    #然后再怕结果排序
    sort=sorted(zip(docs,similarities),key=lambda x:x[1],reverse=True)
    #输出排序
    for doc, score in sort:
        print(f"  {score:.4f}: {doc}")
    
    '''
    输出：
    0.4427: Python is a programming language
    0.3660: JavaScript is used for web development
    0.1625: Machine learning enables AI applications
    0.1273: Deep learning uses neural networks
    0.1144: Cats are popular pets
    
    '''

#使用向量缓存
def embedding_cache():
    text=["hello agent"]
    #创建一个临时文件
    with tempfile.TemporaryDirectory() as tmpDir:
        #定义存储位置
        store=LocalFileStore(root_path=tmpDir)
        #定义向量模型，带缓存版本
        embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
        embedding=CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embedding_model,
            document_embedding_cache=store,
            namespace="openai"
        )

        print(f"第一次向量用openai服务器")
        vect1=embedding.embed_documents(text)
        print(f"embeded{len(vect1)}个document")

        print(f"第二次向量用缓存")
        vect2=embedding.embed_documents(text)
        print(f"embeded{len(vect2)}个document")

        print(f"\nSame vectors:{np.allclose(vect1,vect2)}")    

        '''
        输出这个：
        第一次向量用openai服务器
        embeded1个document
        第二次向量用缓存
        embeded1个document

        Same vectors:True
        
        '''

    






if __name__ == "__main__":
    #Similarity_Search()
    embedding_cache()