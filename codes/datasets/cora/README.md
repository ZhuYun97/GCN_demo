## Introduction of dataset
This directory contains the a selection of the Cora dataset (www.research.whizbang.com/data).

The Cora dataset consists of Machine Learning papers. These papers are classified into one of the following seven classes:
		Case_Based
		Genetic_Algorithms
		Neural_Networks
		Probabilistic_Methods
		Reinforcement_Learning
		Rule_Learning
		Theory

The papers were selected in a way such that in the final corpus every paper cites or is cited by atleast one other paper. There are 2708 papers in the whole corpus. 

After stemming and removing stopwords we were left with a vocabulary of size 1433 unique words. All words with document frequency less than 10 were removed.


THE DIRECTORY CONTAINS TWO FILES:

The .content file contains descriptions of the papers in the following format:

		<paper_id> <word_attributes>+ <class_label>

The first entry in each line contains the unique string ID of the paper followed by binary values indicating whether each word in the vocabulary is present (indicated by 1) or absent (indicated by 0) in the paper. Finally, the last entry in the line contains the class label of the paper.

The .cites file contains the citation graph of the corpus. Each line describes a link in the following format:

		<ID of cited paper> <ID of citing paper>

Each line contains two paper IDs. The first entry is the ID of the paper being cited and the second ID stands for the paper which contains the citation. The direction of the link is from right to left. If a line is represented by "paper1 paper2" then the link is "paper2->paper1". 

---
## The usage of dataset

### 处理cora.content文件
1. 使用np.genfromtxt将content文件中的信息转化成一个array，第一列代表论文的id，中间的列代表论文的feature，最后一列代表论文的类别
data= 
```
array([['31336', '0', '0', ..., '0', '0', 'Neural_Networks'],
       ['1061127', '0', '0', ..., '0', '0', 'Rule_Learning'],
       ['1106406', '0', '0', ..., '0', '0', 'Reinforcement_Learning'],
       ...,
       ['1128978', '0', '0', ..., '0', '0', 'Genetic_Algorithms'],
       ['117328', '0', '0', ..., '0', '0', 'Case_Based'],
       ['24043', '0', '0', ..., '0', '0', 'Neural_Networks']],
      dtype='<U22')
```

2. 使用scipy.sparse.csr_matrix提取出论文的feature（使用压缩稀疏行矩阵构建稀疏矩阵，todense可以还原成普通矩阵）
`features = scipy.sparse.csr_matrix(data[:, 1:-1], dtype=np.float32)`

3. 将label编码成one_hot形式

### 处理cora.cites文件
4. 得到引用关系
```
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

array([[     35,    1033],
       [     35,  103482],
       [     35,  103515],
       ...,
       [ 853118, 1140289],
       [ 853155,  853118],
       [ 954315, 1155073]], dtype=int32)

                                    ```
5. 得到引用关系的邻接矩阵