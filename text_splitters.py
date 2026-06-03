from langchain_text_splitters import(
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
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

















if __name__ == "__main__":
  #recursive_splitter()
  markdown_splitter()