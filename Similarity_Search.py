from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np


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

    






if __name__ == "__main__":
    Similarity_Search()