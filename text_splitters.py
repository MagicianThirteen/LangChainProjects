from langchain_text_splitters import(
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
 
#文本案例
# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''

# 使用递归分割
def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks=splitter.split_text(SAMPLE_TEXT)

    print(f"Original length: {len(SAMPLE_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes:{[len(c) for c in chunks]}")
    print(f"\n First chunk preview:\n{chunks[0][:100]}")
    '''
    输出
    Number of chunks: 2
    Chunk sizes:[462, 423]

    First chunk preview:
    # Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that ena
    '''

#markdown的splitters的使用
def markdown_splitter():
    headers_to_consider=[
        ("#","h1"),
        ("##","h2"),
        ("###","h3"),
    ]
    splitter=MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_consider,
        strip_headers=True,#正文要不要带标题，默认是true
    )
    chunks=splitter.split_text(SAMPLE_TEXT)
    print(f"产生{len(chunks)}个chunk")
    for i,chunk in enumerate(chunks):
       print(f"第{i+1}个chunk的metadata是：{chunk.metadata}")
       print(f"第{i+1}个chunk的content是：{chunk.page_content[:100]}")
'''
输出这个
产生4个chunk
第1个chunk的metadata是：{'h1': 'Introduction to Machine Learning'}
第1个chunk的content是：Machine learning is a subset of artificial intelligence that enables systems to learn and improve fr
第2个chunk的metadata是：{'h1': 'Introduction to Machine Learning', 'h2': 'Types of Machine Learning', 'h3': 'Supervised Learning'}
第2个chunk的content是：Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs
第3个chunk的metadata是：{'h1': 'Introduction to Machine Learning', 'h2': 'Types of Machine Learning', 'h3': 'Unsupervised Learning'}
第3个chunk的content是：Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure wit
第4个chunk的metadata是：{'h1': 'Introduction to Machine Learning', 'h2': 'Applications'}
第4个chunk的content是：Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Reco


'''

#切割代码
def code_splitters():
   python_splitters=RecursiveCharacterTextSplitter.from_language(
      language=Language.PYTHON,
      chunk_size=500,
      chunk_overlap=50,
   )
   chunks=python_splitters.split_text(SAMPLE_CODE)

   for i,chunk in enumerate(chunks):
      print(f"{i+1} chunk chars: {len(chunk)}")
      print(f"{chunk[:150]}..." if len(chunk)>150 else chunk)

'''
输出
1 chunk chars: 390
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr)...
2 chunk chars: 402
def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, r...

'''

#pdf先读取，在分割
def pdf_splitter():
  pdfloader=PyPDFLoader(".\docs\langchain_demo.pdf")
  documents=pdfloader.load()

  #分割
  #注意，这里是先生成分割器，然后用分割器的方式去分割
  pdf_splitter=RecursiveCharacterTextSplitter(
     chunk_size=500,
     chunk_overlap=50,
  )
  chunks=pdf_splitter.split_documents(documents)

  for i,chunk in enumerate(chunks):
     print(f"{i+1} chunk meatada: {chunk.metadata}")
     print(f"{i+1} chunk content: {chunk.page_content}")
'''
输出这样的，后面数据太长了就不贴了
1 chunk meatada: {'producer': 'PyPDF', 'creator': 'PyPDF', 'creationdate': '2026-02-02T21:45:08+00:00', 'source': '.\\docs\\langchain_demo.pdf', 'total_pages': 3, 'page': 0, 'page_label': '1'}
1 chunk content: LangChain Document Loaders - Demo Document
Understanding LangChain Document Loaders
1. Introduction
LangChain provides powerful document loaders that allow you to ingest data from various sources into your
LLM applications. Document loaders are essential for building RAG (Retrieval-Augmented Generation)
systems, chatbots, and knowledge bases.
This document serves as a demo file to test PDF loading capabilities in LangChain. When loaded, this
2 chunk meatada: {'producer': 'PyPDF', 'creator': 'PyPDF', 'creationdate': '2026-02-02T21:45:08+00:00', 'source': '.\\docs\\langchain_demo.pdf', 'total_pages': 3, 'page': 0, 'page_label': '1'}
2 chunk content: content will be split into chunks and can be used for vector storage and retrieval.
2. Types of Document Loaders
  PyPDFLoader: Load PDF files page by page with metadata
  TextLoader: Load plain text files (.txt)
  CSVLoader: Load CSV files with row-based documents
  JSONLoader: Load JSON files with jq-style extraction
  UnstructuredLoader: Handle various file formats automatically
  DirectoryLoader: Load all files from a directory
  WebBaseLoader: Scrape and load web pages

'''















if __name__ == "__main__":
  #recursive_splitter()
  #markdown_splitter()
  #code_splitters()
  pdf_splitter()