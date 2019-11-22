## 目 录

第一部分 理论部分  
- 第1章 深度学习简介 2  
- 第2章 PyTorch环境安装 33  
- 第3章 PyTorch基础知识 40  
- 第4章 简单案例入门 47  
- 第5章 前馈神经网络 59  
- 第6章 PyTorch可视化工具 89  

第二部分 实战部分  
- 第7章 卷积神经网络 110  
- 第8章 循环神经网络简介 145  
- 第9章 自编码模型 164  
- 第10章 对抗生成网络 172  
- 第11章 Seq2seq自然语言处理 186  
- 第12章 利用PyTorch实现量化交易 204  

## 说明

大部分 .py 文件是原书作者的代码。为了更好地可视化，我又用 jupyter notebook 实现了书中的各类模型，并尽可能保证框架一致。主要的改动有：

- [Variable 已经被遗弃了](https://pytorch.org/docs/stable/autograd.html#variable-deprecated)
- 大部分模型都是用 gpu 跑的，在文件名里已注明。不需要的可以删掉代码里的`.cuda()`.
- 对于一些模型结果又增加了可视化的效果。如自编码模型在隐空间的三维分布、对一维插值后生成图片的渐变效果，等等。

欢迎童鞋们对这个版本的代码继续更新提issue~

## 相关链接
[本书链接](http://www.broadview.com.cn/book/5273)
[我对本书的评价](https://baileyswu.github.io/2019/11/pytorch-tutorial/)
