# breast_cancer_data
乳腺癌检测分类数据



## SVM分类器（支持向量机）

1.SVM分类器：完全线性可分情况下的线性分类器，也就是线性可分的情况，是最原始的 SVM，它最核心的思想就是找到最大的分类间隔；

2.硬间隔：完全分类准确，不存在分类错误；
软间隔：允许一定量的样本分类错误；

3.核函数：线性不可分情况下的非线性分类器，引入了核函数。它让原有的样本空间通过核函数投射到了一个高维的空间中，从而变得线性可分。

4.有监督学习和无监督学习的理解：
- 有监督学习和无监督学习的根本区别，就是训练数据中是否有标签。监督学习的数据既有特征又有标签，而非监督学习的数据中只有特征而没有标签。
监督学习是通过训练让机器自己找到特征和标签之间的联系，在以后面对只有特征而没有标签的数据时可以自己判别出标签。
- 非监督学习由于训练数据中只有特征没有标签，所以就需要自己对数据进行聚类分析，然后就可以通过聚类的方式从数据中提取一个特殊的结构。

5.SVM多分类问题
- 一对多法

假设我们要把物体分成 A、B、C、D 四种分类，那么我们可以先把其中的一类作为分类 1，其他类统一归为分类 2。这样我们可以构造 4 种 SVM，分别为以下的情况：

（1） 样本 A 作为正集，B，C，D 作为负集；

（2）样本 B 作为正集，A，C，D 作为负集；

（3）样本 C 作为正集，A，B，D 作为负集；

（4）样本 D 作为正集，A，B，C 作为负集。

- 一对一法

我们可以在任意两类样本之间构造一个 SVM，这样针对 K 类的样本，就会有 C(k,2) 类分类器。

比如我们想要划分 A、B、C 三个类，可以构造 3 个分类器：

（1）分类器 1：A、B；

（2）分类器 2：A、C；

（3）分类器 3：B、C。

当对一个未知样本进行分类时，每一个分类器都会有一个分类结果，即为 1 票，最终得票最多的类别就是整个未知样本的类别。
