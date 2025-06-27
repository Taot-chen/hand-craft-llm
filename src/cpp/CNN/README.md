CNN


## 1 卷积神经网络 CNN 原理

卷积神经网络(Convolutional Neural Networks, CNNs)是一种深度学习算法，特别适用于图像处理和分析，适合用于图像特殊提取和分类。


CNN 由若干层不同类型的网络连接构成，如下图

![alt text](./images/image.png)

输入图像首先经过一次卷积处理，得到卷积层 C1；然后经过一次下采样（池化）处理得到下采样层 S2；之后经过一次卷积处理得到卷积层 C3；C3 经过下采样处理得到下采样层 S4；最后经过全连接处理得到全连接层 C5。至此卷积处理结束，通过卷积处理提取到的特征信息输入到传统的全链接神经网络进行后续处理。


### 1.1 输入层（Input Layer）

网络的输入层，负责接受原始图像数据。


### 1.2 卷积层（Convilutional Layer）

在卷积层中，一组可学习的滤波器（卷积核）在输入数据上滑动，以生成特征图（Feature Maps）。卷积操作允许网络学习到输入数据的局部特征。此外，**由于滤波器的权重在图像的不同部分是共享的，卷积层可以显著减少模型的参数数量，从而减轻过拟合的风险**。

权重共享？卷积层通过卷积的方式提取特征，由于输入图像各个位置使用的卷积核相同，即**提取特征的方式与位置无关**，也就是图像在每一处统计特征的方式都完全相同。这意味着在图像某一处学习的特征，在图像的其他位置也可以使用。换言之，对于输入图像的所有位置，都可以使用相同的学习特征。

卷积层的运算过程如下图，用一个卷积核扫完整张图片：

![alt text](./images/image-2.gif)

卷积的过程可以看做是使用一个过滤器（卷积核）过滤输入图像的各个区域，并从这些区域中提取特征。

在实际使用中，往往有很多个卷积核。一般认为，每个卷积核代表了一种图像模式，如果某个图像块与此卷积核卷积出的值大，则认为此图像块十分接近于此卷积核所能识别的特征。

如果我们设计了6个卷积核，可以理解为我们认为是这个图像上有 6 种深层次的特征模式，我们使用 6 种基础的特征模式来处理输入图像。

![Alt text](./images/image-1.png)

如下图，是一个 25 种不同的卷积核的示例：

![Alt text](./images/image-2.png)



### 1.3 ReLU 层（Rectified Linear Unit Layer）

ReLU 层，非线性操作层，通常跟在卷积层之后，其作用是通过非线性函数来增加网络的非线性特征。



### 1.4 池化层（Pooling Layer）

池化层， 也称作下采样层，其主要功能是降低特征图的空间尺寸，从而降低模型的计算复杂性，并在一定程度上提供模型的平移不变性。

常见的池化操作有最大池化(Max Pooling)和平均池化(Average Pooling)。

当图像太大时，池化层部分将减少参数的数量。空间池化也称为下采样，可在保留重要信息的同时降低数据维度，常见的池化操作有最大池化(Max Pooling)和平均池化(Average Pooling)，以及加和池化(Sum Pooling)。

最常见的是最大值池化，其将输入的图像划分为若干个矩形区域，对每个子区域输出该区域的最大值。这种机制能够有效的原因在于：在发现一个特征之后，它的精确位置远不及它和其他特征的相对位置的关系重要。**池化层会不断地减小数据的空间大小，能够在一定程度上控制过拟合**。通常来说，CNN 的卷积层之间都会周期性地插入池化层。



### 1.5 全连接层（Fully Connected Layer）

在一系列的卷积层和池化层之后，全连接层被用于**对之前提取的特征进行高级别的推理**。在这一层中，**所有的输入都被连接到每个神经元**，这与传统的神经网络类似。这个部分就是最后一步了，经过卷积层和池化层处理过的数据输入到全连接层，得到最终想要的结果。经过卷积层和池化层降维过的数据，全连接层才能**跑得动**，不然数据量太大，计算成本高，效率低下。



### 1.6 输出层（Output Layer）

输出层通常使用`softmax`激活函数进行**多类分类**，或使用`sigmoid`激活函数进行**二分类**。




## 2 反向传播算法（Backpropagation）

### 2.1 DNN 的反向传播算法

在 DNN 中，首先就算出输出层的 $
\delta^L$：

$$\delta^L = \frac{\partial J(W,b)}{\partial z^L} = \frac{\partial J(W,b)}{\partial a^L}\odot \sigma^{'}(z^L)
$$

利用数学归纳法，使用 $\delta^{l+1}$ 的值逐步向前计算出第 $l$ 层的 $\delta^l$，计算表达式：

$$
\delta^{l} = (\frac{\partial z^{l+1}}{\partial z^{l}})^T\delta^{l+1} = (W^{l+1})^T\delta^{l+1}\odot \sigma^{'}(z^l)
$$


有了 $\delta^l$ 的表达式，从而可以求出 $W,b$ 的梯度表达式：

$$
\frac{\partial J(W,b)}{\partial W^l} = \delta^{l}(a^{l-1})^T
$$

$$
\frac{\partial J(W,b,x,y)}{\partial b^l} = = \delta^{l}
$$


有了 $W,b$ 梯度表达式，就可以用梯度下降法来优化 $W,b$, 求出最终的所有 $W,b$ 的值。


在 CNN 中，存在一些不同的地方，不可以直接使用 DNN 的反向传播算法公式。



### 2.2 CNN 的反向传播算法

相比于 DNN 的反向传播算法，想要借用 DNN 的计算方法，需要考虑：

1) 池化层没有激活函数。可以考虑假定池化层的激活函数为 $\sigma(z) = z$，即激活后还是本身，此时池化层的激活函数的导数恒为 1；
2) 池化层在前向传播的时候，会对输入尺寸进行压缩。在反向传播的时候，需要向前反向推导 $\delta^{l-1}$，这个推导方法和DNN不同；
3) 卷积层通过张量卷积，即若干个矩阵卷积求和得到当前层的输出，这个 DNN 不同。DNN 的全连接层是直接进行矩阵乘法得到当前层的输出。因此，卷积层反向传播的时候，上一层的 $\delta^{l-1}$ 递推计算方法会有所不同；
4) 对于卷积层，由于 $W$ 使用的运算是卷积，那么从 $\delta^l$ 推导出该层的所有卷积核的 $W,b$ 的方式也不同。


另外，在 DNN 中，$a_l,z_l$ 都只是一个向量，而 CNN 中的 $a_l,z_l$ 都是是三维的张量，即由若干个输入的子矩阵组成。

最后，需要注意的是，由于卷积层可以有多个卷积核，各个卷积核的处理方法是完全相同且独立的，为了简化算法公式的复杂度，下面提到的卷积核都是卷积层中若干卷积核中的一个。


#### 2.2.1 已知池化层的 $\delta^l$，推导上一隐藏层的 $\delta^{l-1}$

上面提到的四个问题，第一个问题很好解决，这里先解决第二个问题，在一直池化层的 $\delta^l$ 的时候，推导上一隐藏层的 $\delta^{l-1}$。

在前向传播阶段，池化层一般使用 max_pooling 或者 average_pooling，对输入进行下采样，且池化的窗口尺寸已知。现在需要反过来，从缩小后的误差 $\delta^l$，反推前一次较大区域的误差。

在反向传播的阶段：

    * 首先把 $\delta^l$ 的所有子矩阵矩阵大小还原成池化之前的大小；
    * 然后如果是 max_pooling，则把 $\delta^l$ 的所有子矩阵的各个池化窗口的值放在之前做前向传播算法得到最大值的位置；如果是Average_pooling，则把 $\delta^l$ 的所有子矩阵的各个池化窗口的值取平均后放在还原后的子矩阵位置。这个过程一般叫做upsample。

例如，假设池化的窗口尺寸是 `2x2`，且 stride = 2，$\delta^l$ 的第 $k$ 个子矩阵为：

$$
\delta_k^l = \left( \begin{array}{ccc}
    2& 8 \\
    4& 6
\end{array} \right)
$$

那么 $\delta_k^l$ 还原后变成：

$$
\left( \begin{array}{ccc}
    0&0&0&0 \\
    0&2& 8&0 \\
    0&4&6&0 \\
    0&0&0&0
\end{array} \right)
$$

如果是 max_pooling，假设之前在前向传播时记录的最大值位置分别是左上，右下，右上，左下，则转换后的矩阵为：

$$
\left( \begin{array}{ccc}
    2&0&0&0 \\
    0&0& 0&8 \\
    0&4&0&0 \\
    0&0&6&0
\end{array} \right)
$$

如果是Average_pooling，则进行平均，那么转换后的矩阵为：
$$
\left( \begin{array}{ccc}
    0.5&0.5&2&2 \\
    0.5&0.5&2&2 \\
    1&1&1.5&1.5 \\
    1&1&1.5&1.5
\end{array} \right)
$$

这样我们就得到了上一层 $\frac{\partial J (W, b)}{\partial a_k ^{l-1}}$ 的值，那么得到 $\delta_k^{l-1}$：

$$
\delta_k^{l-1} = (\frac{\partial a_k^{l-1}}{\partial z_k^{l-1}})^T\frac{\partial J(W,b)}{\partial a_k^{l-1}} = upsample(\delta_k^l) \odot \sigma^{'}(z_k^{l-1})
$$

其中，upsample 函数完成了池化误差矩阵放大与误差重新分配的逻辑。


$\rightarrow$ 对于张量 $\delta^{l-1}$，有：

$$
\delta^{l-1} = upsample(\delta^l) \odot \sigma^{'}(z^{l-1})
$$


#### 2.2.2 已知卷积层的 $\delta^l$，推导上一隐藏层的 $\delta^{l-1}$

对于卷积层的反向传播，首先考虑卷积层的前向传播公式：

$$
a^l= \sigma(z^l) = \sigma(a^{l-1}*W^l +b^l)
$$

在 DNN 中，$\delta^{l-1}$ 和 $\delta^{l}$ 的递推关系为：

$$
\delta^{l} = \frac{\partial J(W,b)}{\partial z^l} =(\frac{\partial z^{l+1}}{\partial z^{l}})^T \frac{\partial J(W,b)}{\partial z^{l+1}} =(\frac{\partial z^{l+1}}{\partial z^{l}})^T\delta^{l+1}
$$


$\delta^{l-1}$ 和 $\delta^{l}$ 的递推关系中包含 $\frac{\partial z^{l}}{\partial z^{l-1}}$ 的梯度表达式。

注意到 $z^{l}$ 和 $z^{l-1}$ 的关系为：

$$
z^l = a^{l-1}*W^l +b^l =\sigma(z^{l-1})*W^l +b^l
$$

因此有：

$$
\delta^{l-1} = (\frac{\partial z^{l}}{\partial z^{l-1}})^T\delta^{l} = \delta^{l} * rot180(W^{l}) \odot \sigma^{'}(z^{l-1})
$$

这里的式子和DNN的类似，区别在于对于对含有卷积的式子求导时，卷积核被旋转了180度，即式子中的 $rot180()$。**翻转 180 度的含义是上下翻转一次，接着左右翻转一次**。

在 DNN 中这里是矩阵转置。由于这里都是张量，直接推演参数太多。以一个简单的例子说明为什么这里求导后卷积核要翻转。

假设 $l-1$ 层的输出 $a^{l-1}$ 是一个`3x3`矩阵，第 $l$ 层的卷积核 $W^l$ 是一个`2x2`矩阵，采用 1 像素的步幅(stride=1，且不加 pad)，则输出 $z^{l}$ 是一个 2x2 的矩阵（stride=1，Kernel=2，pad=0）。简化 $b^l$ 都是 0, 则有:

$$
a^{l-1}*W^l = z^{l}
$$

列出 $a,W,z$ 的矩阵表达式如下：

$$
\left( \begin{array}{ccc}
    a_{11}&a_{12}&a_{13} \\
    a_{21}&a_{22}&a_{23}\\
    a_{31}&a_{32}&a_{33}
\end{array} \right) * \left( \begin{array}{ccc}
    w_{11}&w_{12}\\
    w_{21}&w_{22}
\end{array} \right) = \left( \begin{array}{ccc}
    z_{11}&z_{12}\\
    z_{21}&z_{22}
\end{array} \right)
$$

利用卷积的定义，很容易得出：

$$
z_{11} = a_{11}w_{11} + a_{12}w_{12} + a_{21}w_{21} + a_{22}w_{22}
$$

$$
z_{12} = a_{12}w_{11} + a_{13}w_{12} + a_{22}w_{21} + a_{23}w_{22}
$$

$$
z_{21} = a_{21}w_{11} + a_{22}w_{12} + a_{31}w_{21} + a_{32}w_{22}
$$

$$z_{22} = a_{22}w_{11} + a_{23}w_{12} + a_{32}w_{21} + a_{33}w_{22}
$$

接着模拟反向求导：

$$
\nabla a^{l-1} = \frac{\partial J(W,b)}{\partial a^{l-1}} = ( \frac{\partial z^{l}}{\partial a^{l-1}})^T\frac{\partial J(W,b)}{\partial z^{l}} =(\frac{\partial z^{l}}{\partial a^{l-1}})^T \delta^{l}
$$

从上式可以看出，对于 $a^{l-1}$ 的梯度误差 $\nabla a^{l-1}$，等于第 $l$ 层的梯度误差乘以 $\frac{\partial z^{l}}{\partial a^{l-1}}$，而 $\frac{\partial z^{l}}{\partial a^{l-1}}$ 对应上面的例子中相关联的 $w$ 的值。

假设 $z$ 矩阵对应的反向传播误差是 $\delta_{11}, \delta_{12}, \delta_{21}, \delta_{22}$ 组成的`2x2`矩阵，则利用上面梯度的式子和 4 个等式，可以分别写出 $\nabla a^{l-1}$ 的9个标量的梯度。

对于 $a_{11}$ 的梯度，由于在 4 个等式中 $a_{11}$ 只和 $z_{11}$ 有乘积关系，从而有：

$$
\nabla a_{11} = \delta_{11}w_{11}
$$

对于 $a_{12}$ 的梯度，由于在 4 个等式中 $a_{12}$ 和 $z_{12}，z_{11}$ 有乘积关系，从而有：

$$
\nabla a_{12} = \delta_{11}w_{12} + \delta_{12}w_{11}
$$

同理可得：

$$
\nabla a_{13} = \delta_{12}w_{12}
$$

$$
\nabla a_{21} = \delta_{11}w_{21} + \delta_{21}w_{11}
$$

$$
\nabla a_{22} = \delta_{11}w_{22} + \delta_{12}w_{21} + \delta_{21}w_{12} + \delta_{22}w_{11}
$$

$$
\nabla a_{23} = \delta_{12}w_{22} + \delta_{22}w_{12}
$$

$$\nabla a_{31} = \delta_{21}w_{21}
$$

$$\nabla a_{32} = \delta_{21}w_{22} + \delta_{22}w_{21}
$$

$$
\nabla a_{33} = \delta_{22}w_{22}
$$

上面 9 个式子可以用一个矩阵卷积的形式表示，即：

$$
\left( \begin{array}{ccc}
    0&0&0&0 \\
    0&\delta_{11}& \delta_{12}&0 \\
    0&\delta_{21}&\delta_{22}&0 \\
    0&0&0&0
\end{array} \right) * \left( \begin{array}{ccc}
    w_{22}&w_{21}\\
    w_{12}&w_{11}
\end{array} \right) = \left( \begin{array}{ccc}
    \nabla a_{11}&\nabla a_{12}&\nabla a_{13} \\
    \nabla a_{21}&\nabla a_{22}&\nabla a_{23}\\
    \nabla a_{31}&\nabla a_{32}&\nabla a_{33}
\end{array} \right)
$$

为了符合梯度计算，在误差矩阵周围填充了一圈 0，此时将卷积核翻转后和反向传播的梯度误差进行卷积，就得到了前一次的梯度误差。



#### 2.2.3 已知卷积层的 $\delta^l$，推导该层的 $W,b$ 的梯度

对于全连接层，可以按 DNN 的反向传播算法求该层 $W,b$ 的梯度，而池化层没有 $W,b$,不用求 $W,b$ 的梯度。只有卷积层的 $W,b$ 需要求出。

意到卷积层 $z$ 和 $W,b$ 的关系为：

$$
z^l = a^{l-1}*W^l +b
$$

因此有：

$$
\frac{\partial J(W,b)}{\partial W^{l}}=a^{l-1} *\delta^l
$$

注意到此时卷积核没有反转，主要是此时是层内的求导，而不是反向传播到上一层的求导:

一个简化的例子，这里输入是矩阵，那么对于第 $l$ 层，某个个卷积核矩阵 $W$ 的导数可以表示如下：

$$
\frac{\partial J(W,b)}{\partial W_{pq}^{l}} = \sum\limits_i\sum\limits_j(\delta_{ij}^la_{i+p-1,j+q-1}^{l-1})
$$

假设输入 $a$ 是`4x4`的矩阵，卷积核 $W$ 是`3x3`的矩阵，输出 $z$ 是`2x2`的矩阵,那么反向传播的 $z$ 的梯度误差 $\delta$ 也是`2x2`的矩阵。

那么根据上面的式子，有：

$$
\frac{\partial J(W,b)}{\partial W_{11}^{l}} = a_{11}\delta_{11} + a_{12}\delta_{12} + a_{21}\delta_{21} + a_{22}\delta_{22}
$$

$$
\frac{\partial J(W,b)}{\partial W_{12}^{l}} = a_{12}\delta_{11} + a_{13}\delta_{12} + a_{22}\delta_{21} + a_{23}\delta_{22}
$$

$$
\frac{\partial J(W,b)}{\partial W_{13}^{l}} = a_{13}\delta_{11} + a_{14}\delta_{12} + a_{23}\delta_{21} + a_{24}\delta_{22}
$$

$$
\frac{\partial J(W,b)}{\partial W_{21}^{l}} = a_{21}\delta_{11} + a_{22}\delta_{12} + a_{31}\delta_{21} + a_{32}\delta_{22}
$$

最终可以整理出这样的九个式子，整理成矩阵形式后可得：

$$
\frac{\partial J(W,b)}{\partial W^{l}} =\left( \begin{array}{ccc} a_{11}&a_{12}&a_{13}&a_{14} \\ a_{21}&a_{22}&a_{23}&a_{24} \\ a_{31}&a_{32}&a_{33}&a_{34} \\
a_{41}&a_{42}&a_{43}&a_{44} \end{array} \right) * \left( \begin{array}{ccc}
\delta_{11}& \delta_{12} \\ \delta_{21}&\delta_{22} \end{array} \right)
$$


对于 $b$，有些特殊，因为 $\delta^l$ 是高维张量，而 $b$ 是一个向量，不能像 DNN 那样直接和 $\delta^l$ 相等。通常的做法是将 $\delta^l$ 的各个子矩阵的项分别求和，得到一个误差向量，即为 $b$ 的梯度：

$$
\frac{\partial J(W,b)}{\partial b^{l}} = \sum\limits_{u,v}(\delta^l)_{u,v}
$$


## 3 C++ 搭建 CNN 网络

### 3.1 CNN 模型封装

`ModelCNN` 类是提供对外部可访问的类，用来构建模型。

```cpp
class ModelCNN {
    private:
        std::vector<layer_t*> layers;
    public:
        ModelCNN () {}

        void conv_layer (uint16_t stride, uint16_t kernel_size, uint16_t num_kernel, td_size in_size);
        void relu_layer (td_size in_size);
        void pool_layer (uint16_t stride, uint16_t kernel_size, td_size in_size);
        void fc_layer (td_size in_size, int out_size);

        td_size& output_size() {return this->layers.back()->out.size;}

        int inference ();
        tensor_t<float>& infer_info() {return this->layers.back()->out;}
        void forward (tensor_t<float>& input);
        float train (tensor_t<float>& input, tensor_t<float>& label);
};
```

首先是构建网络，CNN 模型由若干层网络构成，`ModelCNN` 里的 `vector<layer_t*> layers` 存放指向每一层 `Layer` 的指针，然后增加不同 `Layer` 的接口, 分别是 `conv_layer`，`pool_layer`， `fc_layer`，`relu_layer`，不同的网络层需要的参数不一样，按需要自定义。

在训练部分，一次只能喂一张图片进去，需要提供输入图像和它的 label。训练需要：

* 先将输入数据正向传播一遍然后得到此次的输出（一个10维的向量，即10分类）；
* 将得到的输出再倒序一层一层求偏导反向传播回去；
* 反向传播结束后就得到了每一层输入的偏导，以及每一层卷积核（或权重）的偏导；
* 然后 update 每一层的权重（梯度下降更改权重）；
* 最后计算误差。


```cpp
// func for training
float ModelCNN::train (tensor_t<float>& input, tensor_t<float>& label) {
    // forward
    this->forward(input);
    auto res_info = this->layers.back()->out - label;

    // backward
    for (int i = this->layers.size() - 1; i >= 0; i--)
        this->layers[i]->backward(i < this->layers.size() - 1 ? this->layers[i + 1]->grad_in : res_info);

    // update weights
    for (int i = 0; i < this->layers.size(); i++)
        this->layers[i]->update_weights();

    float err = 0;
    for (int i = 0; i < 10; i++) {
        float res = label(i, 0, 0) - res_info(i, 0, 0);
        err += res * res;
    }
    return sqrt(err) * 100;
}
```

正向传播 `forward` 需要喂一张输入图片进去，然后顺序调用每一层 `Layer` 的 `forward`,

```cpp
// Model forward
void ModelCNN::forward (tensor_t<float>& input) {
    for (int i = 0; i < this->layers.size(); i++) {
        this->layers[i]->forward(i ? this->layers[i - 1]->out : input);
    }
}
```

最后是推理，`inference` 返回正向传播完后得到答案即预测的数字是哪一个，返回最后得到的答案向量。

```cpp
// Model inference
int ModelCNN::inference () {
    int ret = 0;
    // TODO: change the hard parameters to more general code
    for (int i = 0; i < 10; i++) {
        if (this->layers.back()->out(i, 0, 0) > this->layers.back()->out(ret, 0, 0)) ret  = i;
    }
    return ret;
}
```


### 3.2 Layer 基类封装

不同类型的 Layer 处理的算法不同，各自进行的计算和存的变量也不同，用 C++ 的多态可以很方便进行函数的调用以及其他处理。

Layer 基类的虚函数

* `forward` 正向传播
* `backward` 反向传播求梯度
* `update_weights` 更新权重

```cpp
//layer基类
enum class layer_type {
    conv,
    fc,
    relu,
    pool,
    dropout
};

class layer_t {
    public:
        layer_type _type;
        tensor_t<float> grad_in;
        tensor_t<float> in;
        tensor_t<float> out;
        layer_t (layer_type _type_, td_size in_size, td_size out_size):
            _type(_type_),
            in(in_size.x, in_size.y, in_size.z),
            out(out_size.x, out_size.y, out_size.z),
            grad_in(in_size.x, in_size.y, in_size.z)
        {}
        virtual ~layer_t(){}
        virtual void forward(tensor_t<float>& in) = 0;
        virtual void backward(tensor_t<float>& grad_next_layer) = 0;
        virtual void update_weights() = 0;
};
```

### 3.3 卷积层（convolutional layer）

卷积层是 CNN 核心的网络层，输入是三维 tensor ，输出是三维 tensor 。

```cpp
// Conv layer
class ConvLayer: public layer_t {
    public:
        std::vector<tensor_t<float>> kernels;
        std::vector<tensor_t<gradient_t>> kernel_grads;
        uint16_t stride;
        uint16_t kernel_size;

        ConvLayer(uint16_t stride, uint16_t kernel_size, uint16_t num_kernel, td_size in_size);
        td_size map_to_input(td_size out, int z) {return {out.x * this->stride, out.y * stride, z};}

        struct range_t {
            int min_x, min_y, min_z;
            int max_x, max_y, max_z;
        };

        int get_r (float f, int max_v, int lim_min);
        range_t map_to_output(int x, int y);
        void forward (tensor_t<float>& in) override;
        void update_weights () override;
        void backward (tensor_t<float>& grad_next_layer) override;
};
```

在构建 CNN 结构时会 new 一个 `ConvLayer` 的对象，构造函数参数有 `stride`，卷积核大小 `kernel_size`，卷积核数量 `num_kernel`，输入 tensor 的大小 `in_size`。

```cpp
// initialize conv Kernel
conv_weight(i, j, k) = 1.0f / N * rand() / 2147483647.0; //随机的值是有讲究的，这个是CNN常用的卷积核随机初值设置
```

正向传播，直接模拟即可。

反向传播，核心考虑是 `in -> out` 的过程中，每一个 `in` 的变量贡献到不同的 `out` 变量有不同的系数（系数是卷积核里的变量值），所以反向传播时，每个 `in` 变量的 `grad` 等于：

$$
\sum {正向贡献到 out 的 grad} * {贡献系数}
$$

除了 `in` 变量的 `grad` 之外，还有每个卷积核的偏导，计算公式和上面类似，系数是 `in` 中的变量值。所以对于每个 `in` 变量求出它正向传播时贡献的范围，然后反向求梯度即可。

```cpp
// 卷积层反向传播
for (int i = 0; i < this->in.size.x; i++) {
    for (int j = 0; j < this->in.size.y; j++) {
        ConvLayer::range_t rn = this->map_to_output(i, j);
        for (int k = 0; k < this->in.size.z; k++) {
            float total_err = 0;
            // out[i, j, k] -> in[x, y, z] 有贡献的位置
            for (int ini = rn.min_x; ini <= rn.max_x; ini++) {
                int min_x = ini * this->stride;
                for (int  inj = rn.min_y; inj <= rn.max_y; inj++) {
                    int min_y = inj * this->stride;
                    for (int ink = rn.min_z; ink <= rn.max_z; ink++) {
                        // 贡献的系数 -> 第 k 个核作用 out[ i, j, k] 对应的 in 区域，in[x, y, z] 的系数
                        int kk = this->kernels[ink](i - min_x, j - min_y, k);
                        // 系数 * 偏导
                        total_err += kk * grad_next_layer(ini, inj, ink);
                        // kernel grad 同理
                        this->kernel_grads[ink](i - min_x, j - min_y, k).grad += this->in(i, j, k) * grad_next_layer(ini, inj, ink);
                    }
                }
            }
            this->grad_in(i, j, k) = total_err;
        }
    }
}
```

最后是更新权重，每次调 `update_weights` 之前已经反向传播过了，所以 `grad` 已经求过了，可以直接 `SGD` 梯度下降法更新权重。

```cpp
// 卷积层更新梯度
void update_weights() overide {
    for (int n = 0; n < this->kernels.size(); n++) {
        for (int i = 0; i < this->kernel_size; i++) {
            for (int j = 0; j < this->kernel_size; j++) {
                for (int k = 0; k < this->in.size.z; k++) {
                    float& w = this->kernels[n].get(i, j, k);
                    gradient_t& grad = this->kernel_grads[n].get(i, j, k);
                    w = update_weight(w, grad);
                    update_gradient(grad);
                }
            }
        }
    }
}
```


### 3.4 全连接层（fc layer）

全连接层是 CNN 神经网络的最后一层，在实现的时候默认 fc layer 是最后一层，所以在 fc layer 最后要经过 `sigmoid` 函数。

实际上 fc layer 就是一个特殊的卷积层，卷积核大小和输入的大小相等。

```cpp
class FcLayer: public layer_t {
    public:
        std::vector<float> input;
        tensor_t<float> weights;
        std::vector<gradient_t> gradients;

        FcLayer(td_size in_size, int out_size);

        // 铺平后的id
        int id (int x, int y, int z) {
            return z * (this->in.size.x * this->in.size.y) + y * (this->in.size.x) + x;
        }

        // activation, sigmoid
        float act_func(float x);

        // activation derivative
        float act_derv(float x);

        void forward(tensor_t<float>& in) override;
        void update_weights() override;
        void backward(tensor_t<float>& grad_next_layer)override;
};
```


### 3.5 Relu 层（Relu layer）

```cpp
class ReluLayer: public layer_t {
    public:
        ReluLayer(td_size in_size): layer_t(layer_type::relu, in_size, in_size) {}
        void forward(tensor_t<float>& in) override;
        void update_weights() override {}
        void backward(tensor_t<float>& grad_next_layer) override;
};
```



### 3.6 池化层（Pooling Layer）

```cpp
// MaxPool
class PoolLayer: public layer_t {
    public:
        uint16_t stride;
        uint16_t kernel_size;

        PoolLayer(uint16_t _stride, uint16_t _kernel_size, td_size in_size):
            stride (_stride),
            kernel_size(_kernel_size),
            layer_t(
                layer_type::pool,
                in_size,
                {(in_size.x - _kernel_size) / _stride + 1, (in_size.y - _kernel_size) / _stride + 1, in_size.z}
            )
        {}

        td_size map_to_input (td_size out, int z) {
            return {out.x * this->stride, out.y * this->stride, z};
        }

        struct range_t {
            int min_x, min_y, min_z;
            int max_x, max_y, max_z;
        };

        int get_r (float f, int max_v, int lim_min);
        range_t map_to_output (int x, int y);
        void forward (tensor_t<float>& in) override;
        void update_weights () override {}
        void backward (tensor_t<float>& grad_next_layer) override;
};
```



### 3.7 张量实现(tensor)

```cpp
// tensor
typedef struct dims_t {
    int x, y, z;
} td_size;

template<typename T>
class tensor_t {
    public:
        T* data;
        td_size size;
        tensor_t (int _x, int _y, int _z) {
            this->data = new T[_x * _y * _z];
            this->size.x = _x;
            this->size.y = _y;
            this->size.z = _z;
        }

        tensor_t (const tensor_t& another) {
            this->data = new T[another.size.x * another.size.y * another.size.z];
            memcpy(this->data, another.data, another.size.x * another.size.y * another.size.z * sizeof(T));
            this->size = another.size;
        }
};
```

考虑需要的张量维度，训练时图片是单张的喂，mnist 训练集是灰度图片（单通道），那么输入图片是二维的，经过一层多核卷积后就变成三维，后续三维的张量经过池化后输出是三维，经过 relu 后大小不变还是三维，再次经过卷积层，一个卷积核会输出二维的张量，多个卷积核就得到三维的张量，全连接层是在最后一层，输入三维输出一维。

那么需要的张量 tensor 是三维的，支持的操作有：

* 构造一个三维的 tensor

```cpp
tensor_t(int _x, int _y, int _z) {
    data = new T [_x * _y * _z], size = {_x, _y, _z};
}
```

* 使用下标访问某一位置的值

```cpp
T& operator () (int _x, int _y, int _z) {
    return data[_z * (size.x * size.y) + _y * (size.x) + _x];
}
```

* 张量加减

```cpp
tensor_t<T> operator + (tensor_t<T>& another);
tensor_t<T> operator - (tensor_t<T>& another);
```



### 3.8 优化

传统的 Stochastic Gradient Descent（SGD）用于**寻找函数的局部最小值**。SGD 在每次迭代时只选择一个（随机梯度下降）或一小批（小批量梯度下降）样本来估计梯度并更新模型参数。SGD 的更新规则如下：

$$
W = W - \eta \times \Delta L
$$​

其中：

* $W$ 表示模型的参数
* $\eta$ 是学习率，这是一个超参数，用于控制每次参数更新的步长
* $\Delta L$ 是损失函数 L 对模型参数 W 的梯度，这个梯度是通过在一个样本或一小批样本上计算得出的

在传统的 SGD 算法上进行了优化，使用了 Momentum 和 weight decay 两个 trick 优化 SGD 算法。

* Momentum（动量）主要思想是为梯度下降引入一个动量项，权重不仅受当前梯度影响，也受过去梯度影响。具体来说，每次权重更新不仅取决于当前梯度，还取决于过去的权重更新。这种方法可以**帮助优化器更快地越过平坦区域以及某些局部最小值，减少学习的震荡，并更有可能找到全局最小值**。

* Weight Decay（权重衰减）是一种正则化技术，用于**防止模型过拟合**。在训练过程中，权重衰减会对模型的权重参数进行惩罚，这通常通过在损失函数中添加一个正则化项来实现。这个正则化项是模型权重的 L2 范数（平方和）与一个衰减系数的乘积。通过这种方式，**权重衰减倾向于使模型的权重尽可能小，从而减少模型复杂度，并提高其泛化能力**。

同时使用这两种技术可以更快地收敛训练过程，减少过拟合，提高模型的泛化能力。总的来说，Momentum 可以帮助 SGD 更快地收敛，而 Weight Decay 可以帮助防止模型过拟合。数学形式如下：

* 速度更新：$v = \gamma \times v - \eta \times ( \Delta L + \lambda * W )$
* 权重更新：$W = W + v$

其中：

* $v$ 是速度变量（动量项）
* $\gamma$ 是动量系数
* $\eta$ 是学习率
* $\Delta L$ 是损失函数的梯度
* W 是模型权重
* $\lambda$ 是权重衰减系数


在代码实现上的表现：

```cpp
static float update_weight (float w, gradient_t& grad, float multp = 1) {
    w -= LEARNING_RATE * (grad.grad + grad.pregrad * MOMENTUM) * multp + LEARNING_RATE * WEIGHT_DECAY * w;
    return w;
}

static void update_gradient (gradient_t& grad) {
    grad.pregrad = (grad.grad + grad.pregrad * MOMENTUM);
}
```


**TODO：**
- [ ] 速度优化，使用适量计算指令做矢量化
- [ ] 支持多通道输入
- [ ] 去掉模型参数的 hard code
