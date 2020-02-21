# My Paper Reading List

## Meta learning

- **MAML-based**
    - (**MAML**) Chelsea Finn, Pieter Abbeel, et al. "**Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**." (**2017**). [[arXiv:1703.03400](https://arxiv.org/abs/1703.03400)] :star:
    - (**Reptile**)Alex Nichol, Joshua Achiam, et al. "**On First-Order Meta-Learning Algorithms**."(**2017**). [[arXiv:1803.02999](https://arxiv.org/abs/1803.02999)] :star:

- **Metric-base**
    - (**Siamese Network**) Gregory Koch, et al. "**https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf**."  [[arXiv](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)] 
    - (**Prototypical Networks**) Jake Snell, Kevin Swersky, et al. "**Prototypical Networks for Few-shot Learning**."  (**2017**). [[arXiv:1703.0517](https://arxiv.org/abs/1703.05175)]
    - (**Match Networks**) "**Matching Networks for One Shot Learning**." (**2017**)[[arXiv:1606.04080](https://arxiv.org/abs/1606.04080)]
    - (**Relation Network**) Flood Sung, Yongxin Yang, et al."**Learning to Compare: Relation Network for Few-Shot Learning**" (**CVPR 2018**)[[arXiv:1711.06025](https://arxiv.org/abs/1711.06025)]
    
- **Gradient Descent as LSTM**
    - S Ravi, et al. "**OPTIMIZATION AS A MODEL FOR FEW-SHOT LEARNING**",(**2016**).[[openreview](https://openreview.net/pdf?id=rJY0-Kcll)]
    - "**learning to learn by gradient by gradient**"(**2017**)[[NIPS 2017](https://papers.nips.cc/paper/6461-learning-to-learn-by-gradient-descent-by-gradient-descent.pdf)]


## Metric learning

-  **2048 Like Games**
    - Szubert, Marcin, and Wojciech Jaśkowski. "**Temporal difference learning of n-tuple networks for the game 2048**." Computational Intelligence and Games (CIG), IEEE Conference on. IEEE, (**2014**).
    - Wu, I-Chen, et al. "**Multi-stage temporal difference learning for 2048**." Technologies and Applications of Artificial Intelligence. Springer, Cham, (**2014**).
    - Yeh, Kun-Hao, et al. "**Multi-stage temporal difference learning for 2048-like games**." IEEE Transactions on Computational Intelligence and AI in Games (**2016**).
    - Jaskowski, Wojciech. "**Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping**." IEEE Transactions on Computational Intelligence and AI in Games (**2017**). :star:
- **MCTS**
    - (**UCS**)Brügmann, Bernd. "**Monte carlo go**". Vol. 44. Syracuse, NY: Technical report, Physics Department, Syracuse University, (**1993**).
    - (**UCB**)Auer, Peter, Nicolo Cesa-Bianchi, and Paul Fischer. "**Finite-time analysis of the multiarmed bandit problem.**" Machine learning 47.2-3 (**2002**): 235-256.
    - (**UCT**)Kocsis, Levente, and Csaba Szepesvári. "**Bandit based monte-carlo planning**." European conference on machine learning. Springer, Berlin, Heidelberg, **2006**.
    - (**MCTS**)Coulom, Rémi. "**Efficient selectivity and backup operators in Monte-Carlo tree search**." International conference on computers and games. Springer, Berlin, Heidelberg, **2006**.    
    - (**RAVE**)Gelly, Sylvain, and David Silver. "**Monte-Carlo tree search and rapid action value estimation in computer Go**." Artificial Intelligence 175.11 (**2011**): 1856-1875.
    - Gelly and David. "**Combining online and offline knowledge in UCT**." ICML 2007.
        - ICML 2017: Test of Time Award
    - Chaslot, Guillaume MJ-B. "**Parallel monte-carlo tree search**." International Conference on Computers and Games. Springer, Berlin, Heidelberg, (**2008**).
    - Segal, Richard B. "**On the scalability of parallel UCT**." International Conference on Computers and Games. Springer, Berlin, Heidelberg, (**2010**).
    - Browne, Cameron B., et al. "**A survey of monte carlo tree search methods**." IEEE Transactions on Computational Intelligence and AI in games 4.1 (**2012**): 1-43.
- **AlphaGo** 
    -  Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." [Nature 529.7587](https://www.nature.com/articles/nature16961) (**2016**): 484-489. :star:
        -  APV-MCTS
    -  Silver, David, et al. "**Mastering the game of go without human knowledge**." [Nature 550.7676](https://www.nature.com/articles/nature24270) (**2017**): 354. :star:
    -  Silver, David, et al. "**Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm**."  (**2017**). [[arXiv:1712.01815](https://arxiv.org/abs/1712.01815)] :star:
    -  Silver, David, et al. "**A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play**." [Science 362.6419](http://science.sciencemag.org/content/362/6419/1140) (**2018**): 1140-1144.
- **More**
    - Silver, David, Richard S. Sutton, and Martin Müller. "**Temporal-difference search in computer Go**." Machine learning 87.2 (**2012**): 183-219.
    - Lai, Matthew. "**Giraffe: Using deep reinforcement learning to play chess**." (**2015**). [[arXiv:1509.01549](https://arxiv.org/abs/1509.01549)] 
    -  Vinyals, Oriol, et al. "**StarCraft II: a new challenge for reinforcement learning**." (**2017**). [[arXiv:1708.04782](https://arxiv.org/abs/1708.04782)]    
    -  Maddison, Chris J., et al. "**Move evaluation in go using deep convolutional neural networks**." (**2014**). [[arXiv:1412.6564](https://arxiv.org/abs/1412.6564)]
    -  Soeda, Shunsuke, Tomoyuki Kaneko. "**Dual lambda search and shogi endgames**." Advances in Computer Games. Springer, Berlin, Heidelberg, (**2005**).
    -  (**Darkforest**)Tian, Yuandong, and Yan Zhu. "**Better computer go player with neural network and long-term prediction**." arXiv:1511.06410 (**2015**).
    -  Cazenave, Tristan. "**Residual networks for computer Go.**" _IEEE Transactions on Games_ 10.1 (**2018**): 107-110.
    -  Gao, Chao, Martin Müller, and Ryan Hayward. "**Three-Head Neural Network Architecture for Monte Carlo Tree Search**." IJCAI. **(2018)**.
    -  (**ELF**)Tian, Yuandong, et al. "**Elf: An extensive, lightweight and flexible research platform for real-time strategy games**." Advances in Neural Information Processing Systems. (**2017**).
    - (**ELF2**)Tian, Yuandong, et al. "**ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero**." arXiv:1902.04522 (**2019**).



## NLP



## tips of trick


## Convolutional Neural Network

- (**LeNet**) LeCun, Yann, et al. "**Gradient-based learning applied to document recognition**." Proceedings of the IEEE 86.11 (**1998**).
- (**AlexNet**) Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. (**2012**).
- (**ZFNet**) Zeiler, Matthew D., and Rob Fergus. "**Visualizing and understanding convolutional networks**." European conference on computer vision. Springer, Cham, (**2014**).
- (**NIN**) Lin, Min, Qiang Chen, and Shuicheng Yan. "**Network in network**." (**2013**). [[arXiv:1312.4400](https://arxiv.org/abs/1312.4400)]
- (**VGGNet**) Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**."(2014). [[arXiv:1409.1556](https://arxiv.org/abs/1409.1556)]
- (**GoogLeNet**) Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
- (**BN**) Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." International Conference on Machine Learning. (**2015**). [[arXiv:1502.03167](https://arxiv.org/abs/1502.03167)]
- (**ResNet**) He, Kaiming, et al. "**Deep residual learning for image recognition**." Proceedings of the IEEE conference on computer vision and pattern recognition. (**2016**). [[arXiv:1512.03385](https://arxiv.org/abs/1512.03385)]  [CVPR 2016 Best Paper] :star:
- (**Pre-active**) He, Kaiming, et al. "**Identity mappings in deep residual networks**." European Conference on Computer Vision. Springer International Publishing. (**2016**). [[arXiv:1603.05027](https://arxiv.org/abs/1603.05027)]
- Huang, Gao, et al. "**Deep networks with stochastic depth.**" European Conference on Computer Vision. Springer, Cham, 2016. [[arXiv:1603.09382](https://arxiv.org/abs/1603.09382)]
- (**Wide ResNet**) Zagoruyko, Sergey, and Nikos Komodakis. "**Wide residual networks**." (**2016**). [[arXiv:1605.07146](https://arxiv.org/abs/1605.07146)]
- (**ResNeXt**) Xie, Saining, et al. "**Aggregated residual transformations for deep neural networks**." 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, (**2017**). [[arXiv:1611.05431](https://arxiv.org/abs/1611.05431)]
- (**DenseNet**) Huang, Gao, et al. "**Densely connected convolutional networks**." (**2016**). [[arXiv:1608.06993](https://arxiv.org/abs/1608.06993)] 
- Pleiss, Geoff, et al. "**Memory-efficient implementation of densenets**." arXiv preprint (**2017**). [[arXiv:1707.06990](https://arxiv.org/abs/1707.06990)]
- (**DPN**) Chen, Yunpeng, et al. "**Dual path networks**." Advances in Neural Information Processing Systems. (**2017**). [[arXiv:1707.01629](https://arxiv.org/abs/1707.01629)]
- (**SENet**) Hu, Jie, Li Shen, and Gang Sun. "**Squeeze-and-excitation networks**." (**2017**). [[arXiv:1709.01507](https://arxiv.org/abs/1709.01507)]
- (**CondenseNet**) Huang, Gao, et al. "**CondenseNet: An Efficient DenseNet using Learned Group Convolutions**." (**2017**). [[arXiv:1711.09224](https://arxiv.org/abs/1711.09224)] 
- (**GN**) Yuxin Wu, Kaiming He. "**Group Normalization**." (**2018**). [[arXiv:1803.08494](https://arxiv.org/abs/1803.08494)]


## Optimizers

- Kingma, Diederik P., and Jimmy Ba. "**Adam: A method for stochastic optimization.**" arXiv preprint (2014). [[arXiv:1412.6980](https://arxiv.org/abs/1412.6980)]
- Ruder, Sebastian. "**An overview of gradient descent optimization algorithms.**" arXiv preprint (2016). [[arXiv:1609.04747](https://arxiv.org/abs/1609.04747)]
- Keskar, Nitish Shirish, and Richard Socher. "**Improving Generalization Performance by Switching from Adam to SGD.**" arXiv preprint (2017). [[arXiv:1712.07628](https://arxiv.org/abs/1712.07628)]
- Loshchilov, Ilya, and Frank Hutter. "**SGDR: stochastic gradient descent with restarts**." arXiv preprint (2016). [[arXiv:1608.03983](https://arxiv.org/abs/1608.03983)] :star:
- Smith, Leslie N. "**Cyclical learning rates for training neural networks.**" Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017. [[arXiv:1506.01186](https://arxiv.org/abs/1506.01186)]
- Gastaldi, Xavier. "**Shake-shake regularization.**" arXiv preprint (**2017**). [[arXiv:1705.07485](https://arxiv.org/abs/1705.07485)]
- Huang, Gao, et al. "**Snapshot ensembles: Train 1, get M for free**." arXiv preprint (**2017**). [[arXiv:1704.00109](https://arxiv.org/abs/1704.00109)]
- Jaderberg, Max, et al. "**Population based training of neural networks**." arXiv preprint (**2017**). [[arXiv:1711.09846](https://arxiv.org/abs/1711.09846)] 


## Generative Adversarial Network

- Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in neural information processing systems. (**2014**). [[arXiv:1406.2661](https://arxiv.org/abs/1406.2661)]
- Mirza, Mehdi, and Simon Osindero. "**Conditional generative adversarial nets**." (**2014**). [[arXiv:1411.1784](https://arxiv.org/abs/1411.1784)]
- Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." (**2015**). [[arXiv:1511.06434](https://arxiv.org/abs/1511.06434)]
- Reed, Scott, et al. "**Generative adversarial text to image synthesis**." (**2016**). [[arXiv:1605.05396](https://arxiv.org/abs/1605.05396)]
- Shrivastava, Ashish, et al. "**Learning from simulated and unsupervised images through adversarial training**."(**2016**). [[arXiv:1612.07828](https://arxiv.org/abs/1612.07828)]
- Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "**Wasserstein gan**." (**2017**). [[arXiv:1701.07875](https://arxiv.org/abs/1701.07875)]


## Others 

- Li, Yuxi. "**Deep reinforcement learning: An overview**." (**2017**). [arXiv:1701.07274](https://arxiv.org/abs/1701.07274)
- [AdversarialNetsPapers](https://github.com/zhangqianhui/AdversarialNetsPapers)
- [deep-reinforcement-learning-papers](https://github.com/junhyukoh/deep-reinforcement-learning-papers)
- [BIGBALLON/cifar-10-cnn](https://github.com/BIGBALLON/cifar-10-cnn)
- [aymericdamien/TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)
- [openai/baselines](https://github.com/openai/baselines)
- [rlcode/reinforcement-learning](https://github.com/rlcode/reinforcement-learning)
- [Theory of Computer Games](http://www.iis.sinica.edu.tw/~tshsu/tcg/index.html)


