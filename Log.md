# 100 Days Of ML - LOG

## Day 0 : Jul 6, 2018
 
**Today's Progress** : I'm still struggling to get equipped myself with things to train the network including backprop and gradient descent to give a natural presentation to my team members on the seminar early next week.  
Here are what I've gone through so far
* [Machine Learning Week 5](https://www.coursera.org/learn/machine-learning/home/week/5)
* [Neural Networks and Deep Learning Week 2](https://www.coursera.org/learn/neural-networks-deep-learning/home/week/2)
* [CS231n Lecture 3 - Loss Functions and Optimization](https://www.youtube.com/watch?v=h7iBpEHGVNc&index=3&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
* [CS231n Lecture 4 - Introduction to Neural Networks](https://www.youtube.com/watch?v=d14TUNcbn1k&index=4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
* [CS231n Lecture 6 - Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M&index=6&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
* [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)

**Thoughts** : Copying whole words in the lecture video to my brain and just speak of that as is may be the best method. (if I could)

## Day 1 : Jul 7, 2018

**Today's Progress** : I didn't do much today. Shopping with my wife and son, I planned to do some exercises to make the NN concepts concrete in my head. Took some ML lectures from YouTube. Also, took some RL videos to have more "[Integrative Complexity](https://medium.com/the-mission/studies-show-that-people-who-have-high-integrative-complexity-are-more-likely-to-be-successful-443480e8930c)".

**Thoughts** : CS231n assignments would be great.

## Day 2 : Jul 8, 2018

**Today's Progress** : Coded some graph architecture in Python with Pythonista(iOS App), and took some relative lectures from YouTube.

**Thoughts** : The way Andrej, Justin, and Serena take to explain the backprop is really intuitive and great.

**Link of Work** : [CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1](https://www.youtube.com/watch?v=i94OvYb6noo)

## Day 3 : Jul 9, 2018

**Today's Progress** : Tried tf-pose-estimation with the team members, and had some conversation on it. 

**Thoughts** : It looks pretty promising. But when it comes to further investments, I have to first learn the way playing around it with no hurdles. Such as I/O parts of the framework.

**Link of Work** : [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)

## Day 4 : Jul 10, 2018

**Today's Progress** : Had my own seminar on Machine Learning Fundamental for the team members.

**Thoughts** : That was the 4th series on Machine Learning seminar. (0. Machine Learning Overview, 1. Deep Learning for Computer Vision, 2. Machine Learning Fundamental, 3. Machine Learning Fundamental (cont.)). I think these series are enough for the team members in elevating their mind to this field. The missing parts might be discovered by the team members themselves.

**Link of Work** :  
[Seminar:Computer Vision](https://docs.google.com/presentation/d/1DIe8pH1H1zohX4l4D5ExGoErPxqlfjdazmRXmmXj51Q/edit?usp=sharing)  
[Seminar:Machine Learning Fundamental](https://docs.google.com/presentation/d/1kZGnigyeN1H7z-ohX67aTkAk85CL_qaU6TNhd5VC_a8/edit?usp=sharing)  
I couldn't cite the sites, the lecture, and the papers enough, so please let me know if there's any possible problem. I'll make a change on that.

## Day 5 : Jul 11, 2018

**Today's Progress** : Took some RL lectures.

**Thoughts** : Just had an overview. I feel I should dig into those soon.

**Link of Work** :  
[RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=2pWv7GOvuf0&t=1s)  
[RL Course by David Silver - Lecture 2: Markov Decision Process](https://www.youtube.com/watch?v=lfHX2hHRMVQ)

## Day 6 : Jul 12, 2018

**Today's Progress** : Watched some RL articles and videos.

**Thoughts** : RL is somewhat doens't have dependencies with current project so it can stand parallel. Rather, RL could help one of the other internal projects greatly, I wish.

## Day 7 : Jul 13, 2018

**Today's Progress** : Watched some RL videos.

**Thoughts** : To cover the whole fundamental of RL would take long time. MDP, policy iteration, value iteration, Q-learning, etc.

## Day 8 : Jul 14, 2018

**Today's Progress** : Just watched some RL videos.

**Thoughts** : Got wet some more.

## Day 9 : Jul 15, 2018

**Today's Progress** : Watched some RL videos.

**Thoughts** : More and more.

## Day 10 : Jul 16, 2018

**Today's Progress** : I went further into Realtime Multi-Person Pose Estimation paper, while digging into one of the implemented code simultaneously.

**Thoughts** : 

**Link of Work** :  
[Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1611.08050v2.pdf)
![Network Architecture](https://github.com/ZheC/Multi-Person-Pose-Estimation/raw/master/readme/arch.png)

## Day 11 : Jul 17, 2018

**Today's Progress** : Compared some of the implementation of the pose estimation.

**Thoughts** : I feel the keras implementation is rather clear and robust than the tensorflow implementation.

**Link of Work** :  
[Tensorflow version](https://github.com/ildoonet/tf-pose-estimation)  
[Keras version](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)

## Day 12 : Jul 18, 2018

**Today's Progress** : Stuck to the pose estimation.

**Thoughts** : IMO, the author of tf-pose-estimation seems like a geek. tf-pose-estimation has a lot of multi-gpu codes which embarrass me, making it hard to figure out what a code block is doing. Also it has a lot of hardcoded variables which makes it less flexible.

## Day 13 : Jul 19, 2018

**Today's Progress** : Read the paper again and played around the code.

**Thoughts** : After I grasp the entire pipeline I wondered how the architecture is contructed with tf or keras code. Especially the part about the 6-staged network and about the initialization of the confidence map and the PAFs. Fortunately, today's team meeting has given the answers to those my questions.

## Day 14 : Jul 20, 2018

**Today's Progress** : I had my 4th Seminar about CNN architectures.

**Thoughts** : Even after finishing the online courses, people can easily overlook the basic convolution operations and the propagation through conv layers. Of course, we won't invent the wheel but, knowing the basics can give powers on making breakthrough later.

**Link of Work** : [Seminar: Base Model](https://docs.google.com/presentation/d/14G5X1aNWSlpFz8cVQj-s87A26sTpvWIXLplMka6Gr_g/edit?usp=sharing)

## Day 15 : Jul 21, 2018

**Today's Progress** : I refined one of my Jupyter notebooks which handles the list of young people I should shepherd in the church. That mainly uses pandas and matplotlib to categorize and to visualize the statistics of the categories.

**Thoughts** : Visualizing is always fun!

## Day 16 : Jul 22, 2018

**Today's Progress** : Took an easy-explaned LSTM lecture

**Thoughts** : Now it's clear why LSTM should come that supplements the vanilla RNN.

**Link of Work** :  
[Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)](https://www.youtube.com/watch?v=WCUNPb-5EYI)  
[CS231n Lecture 10 | Recurrent Neural Networks](https://youtu.be/6niqTuYFZLQ)

## Day 17 : Jul 23, 2018

**Today's Progress** :  
1) Dug deep into Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields paper.  
2) I just made a plan to build up a side project, which is named Label4CV. It has only its architecture yet.

**Thoughs** : Label4CV is what I have been having in mind through time. I think that would be fun, and also practical.

**Link of Work** : https://github.com/sungwonida/label4cv

## Day 18 : Jul 24, 2018

**Today's Progress** : Took a RL video, and read about deploying.

**Thoughts** : Deploying models on the edge devices is really a matter. Should have the optimal trade-off among the accuracy and the speed. Maybe some of the solutions(frameworks) that would captivate me are Tensorflow Lite and Core ML.

**Link of Work** :  
[MIT 6.S094: Deep Reinforcement Learning for Motion Planning](https://youtu.be/QDzM8r3WgBw)  
[Machine Learning and Mobile: Deploying Models on The Edge](https://blog.algorithmia.com/machine-learning-and-mobile-deploying-models-on-the-edge/?utm_medium=email&utm_source=topic+optin&utm_campaign=awareness&utm_content=20180723+ai+nl&mkt_tok=eyJpIjoiTkdJd1lXVXdPVE16WXpWaiIsInQiOiJ3MWRLT3M2KzNPUEdLSyt3UGlleEx0MDhZTmVzXC9ZNjdcLzl6RVlQV0UwallveVA5Rkl0SlF0ZUYwNTBWQXJQZGpxMEhJVE94a0h1Mmt5Nkl2d3hvMGF1YWRDYjk5UzZjMmhLRXN2R3JZb3M5K2RtR2N4M1RRalwvZjFTbXJ6TEEyWiJ9)

## Day 19 : Jul 25, 2018

**Today's Progress** : Reading through a paper, "DenseCap: Fully Convolutional Localization Networks for Dense Captioning". No hands-on today.

**Thoughts** : I need some nice architecture being able to analyse the behavior from video. I know there are some papers for that. Start from DenseCap paper, I'll read others through.

**Link of Work** : https://arxiv.org/pdf/1511.07571v1.pdf

## Day 20 : Jul 26, 2018

**Today's Progress** : Kept reading DenseCap. Attention models emerged to be one of the next learning items. And took some lectures on basic object detection from deeplearning.ai

**Thoughts** : While have been focused on relatively high level applications these days, suddenly, came to examine myself by thinking of the basics. Am I good enough now? If not, back to basic. They should not be overlooked. 

## Day 21 : Jul 27, 2018

**Today's Progress** : Test run of the pose estimation on GTX1070. Comparing with the training on the CPU, now it is running with x50 speed! (On the basis of ETA)

**Thoughts** : When considering of my future own machine for deep learning, the selective range has been much larger! Because the only choice of mine was GTX1080ti. I'm pretty happy with GTX1070 now.

**Link of Work** : https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation

## Day 22 : Jul 28, 2018

**Today's Progress** : Took lectures on RNN from deeplearning.ai

**Thoughts** : Prof. Andrew Ng is indeed top in his way of solving the story. I'm getting a ton.

## Day 23 : Jul 29, 2018

**Today's Progress** : Took more lessons on various types of RNNs and some of word embeddings.

**Thoughts** : Um.. actually (I think) I need Attention Models. Maybe I can fastforward to it for now. (deeplearning.ai)

## Day 24 : Jul 30, 2018

**Today's Progress** : Stuck with Keras callbacks for a while especially on ModelCheckpoint. Then engineered on it to work as I thought. Also I needed to get some basics on COCO APIs. So, I went throught the tutorial they provide.

**Thoughts** : At the very first, ModelCheckpoint seemed weird in its behavior. But I noticed soon that I misunderstood the arguments ModelCheckpoint get. 'mode' overwrites 'save_best_only'.

**Link of Work** : https://github.com/cocodataset/cocoapi

## Day 25 : Jul 31, 2018

**Today's Progress** : Made a custom Keras callback based on ModelCheckpoint to add additional function. (max_to_keep)

**Thoughts** : I've been awaring of max_to_keep that of tf.train.Saver. But I surprised at Keras's lack of that option.

**Link of Work** : https://github.com/sungwonida/keras_custom_callbacks

## Day 26 : Aug 1, 2018

**Today's Progress** : Played around Keras Callbacks. With adding some LambdaCallback and custom callbacks I could able to see some logs give me confidence that the model load/save works properly.

**Thoughts** : Keras Callbacks is interesting! Though I should get wet more on Tensorflow itself to be able to implement more useful custom callbacks.

## Day 27 : Aug 2, 2018

**Today's Progress** : Picked up and read through a paper that tells me about activity recognition from video. 

**Thoughts** : Many-to-one RNN model is what I've been searching for activity recognition.

**Link of Works** : http://arxiv.org/abs/1703.10667v1

## Day 28 : Aug 3, 2018

**Today's Progress** : Read through TS-LSTM paper and quickly tried one of the implementations of the base module, two stream convnet, from github.

**Thoughts** : Even the few that existed were not implemented by Tensorflow or Keras. The authors tend to code in Torch7 or PyTorch which I'm not familier yet.

**Link of Works** : https://github.com/jeffreyhuang1/two-stream-action-recognition

## Day 29 : Aug 4, 2018

**Today's Progress** : LSTMs became clear in theory.

**Thoughts** : Need coding practice.

**Link of Works** : http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Day 30 : Aug 5, 2018

**Today's Progress** : Played around Karpathy's raw implementation of RNN. 

**Thoughts** : I want to be able to replicate this code line by line.

**Link of Works** : https://gist.github.com/karpathy/d4dee566867f8291f086

## Day 31 : Aug 6, 2018

**Today's Progress** : Investigated Karpathy's RNN code line by line.

**Thoughs** : Should dig into more cases other than 'hello'.


## Day 32 : Aug 7, 2018

**Today's Progress** : Coding practice by implementating the CNN in Tensorflow and Keras.

**Thoughts** : I should be balanced in between theory and coding skill. Keras functional API seems more appealing to me than the other one.

**Link of Works** :  
http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/  
http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/

## Day 33 : Aug 8, 2018

**Today's Progress** : Coding practice by following the tutorial on the LSTMs in Tensorflow.

**Thoughts** : It's fun, but a little bit confusing to follow the implementation. Had spent much time on looking into the data preparation part.

**Link of Works** : http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/

## Day 34 : Aug 9, 2018

**Today's Progress** : Worked hard on preparing the seminar of mine which will be held tomorrow.

**Thoughts** : Action recognitions using deep learning sort are really interesting.

**Link of Works** :  
[Two-strram Convolutional Networks for Action Recognition](http://arxiv.org/abs/1406.2199v2)  
[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](http://arxiv.org/abs/1608.00859v1)  
[TS-LSTM and Temporal-Inception: Exploiting Spatiotemporal Dynamics for Activity Recognition](http://arxiv.org/abs/1703.10667v1)

## Day 35 : Aug 10, 2018

**Today's Progress** : Seminar time! with the team members. Todays topic was Pose Estimation and Action Recognition.

**Thoughts** : I think I did better than I was worried. Though LSTM should be explained at the next seminar.

**Link of Works** : [Seminar: Pose Estimation + Action Recognition](https://www.google.com/url?q=https://docs.google.com/presentation/d/1uj0K_bCNaY3nDmvH19m7PzYS9a05BERCMCtsAu9-nII/edit?usp%3Dsharing&sa=D&source=hangouts&ust=1533973489946000&usg=AFQjCNGTAWHRal6Yeyjha7macFSYVL-Wgw)

## Day 36 : Aug 11, 2018

**Today's Progress** : Scribbled RNN code in numpy for better understanding.

**Thoughts** : The backprop part is slightly confusing.

## Day 37 : Aug 12, 2018

**Today's Progress** : Practice 1) LSTM coding 2) Python tricks for better understanding and better self-implementations

**Thoughts** : Python tricks is fun.

**Link of Works** :  
https://gist.github.com/karpathy/587454dc0146a6ae21fc  
https://dbader.org/blog/announcing-python-tricks-the-book

## Day 38 : Aug 13, 2018

**Today's Progress** : Practice on Python decorators (while reading the raw LSTM code)

**Thoughts** : Python functions are first-class objects. With continuing this idea, decorators give powerful and flexible way to allow reusable building blocks. I was attracted to Python even more.

## Day 39 : Aug 14, 2018

**Today's Progress** : Looked into the Two-stream ConvNet written in PyTorch

**Thoughts** : Thanks Jedi! (emacs package)

**Link of Works** : https://github.com/jeffreyhuang1/two-stream-action-recognition

## Day 40 : Aug 15, 2018

**Today's Progress** : Followed some tutorials on NLP basic

**Thoughts** : I really wanna be a master on sequence models

## Day 41 : Aug 16, 2018

**Today's Progress** :  
1) Bag-of-words tutorial using sklearn and Keras
2) Word Embeddings using Gensim

**Thoughts** : Jason Brownlee's crash courses are awesome. I need to follow those closely to speed up.

## Day 42 : Aug 17, 2018

**Today's Progress** : Took an RNN lecture from CS231n again and again

**Thoughts** : I should copy the words of Justin into my brain (to better serve my own seminar)

**Link of Works** : https://youtu.be/6niqTuYFZLQ

## Day 43 : Aug 18, 2018

**Today's Progress** : Got some idea on batch normalization  

**Thoughts** : BN is really cool. And it sounds reasonable that Dropout is losing its position in CNN.

**Link of Works** :  
[Don't Use Dropout in Convolutional Networks](https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81486c823c16?source=linkShare-d73d583291ce-1534593369)  
[Intuit and Implement: Batch Normalization](https://towardsdatascience.com/intuit-and-implement-batch-normalization-c05480333c5b?source=linkShare-d73d583291ce-1534593323)

## Day 44 : Aug 19, 2018

**Today's Progress** : Watched RNN lecture(CS231n)

**Thoughts** : I'll watch the video over and over again

**Link of Works** :  
https://youtu.be/yCC09vCHzF8  
https://youtu.be/6niqTuYFZLQ

## Day 45 : Aug 20, 2018

**Today's Progress** : Examined the LSTM code which has been written in numpy by Karpathy. (lines before bt)

**Thoughts** : Those lines are better understandable when come along with cs231n. (lecture 10)

**Link of Works** : https://gist.github.com/karpathy/587454dc0146a6ae21fc

## Day 46 : Aug 21, 2018

**Today's Progress** : Examined the rest of the LSTM code. (from bt)

**Thoughts** : I'm happy that I could stand on the shoulders of giants like Karpathy.

## Day 47 : Aug 22, 2018

**Today's Progress** : I have glimpsed at attention mechanism. Plus, reminded the recurrent neural networks.

**Thoughts** : I suddenly turned my attention to attention mechanism, not because did I mastered the LSTMs, but I just wanted to follow up the SOTA trand in sequence modeling. Since one of my biggest interest in ML is edge computing.

**Link of Works** :  
[The fall of RNN / LSTM](https://justread.link/u9Jv4QAcr)  
[LSTM Networks - The Math of Intelligence (Week 8)](https://youtu.be/9zhrxE5PQgY)  
[An Introduction to LSTMs in Tensorflow](https://youtu.be/l4X-kZjl1gs)

## Day 48 : Aug 23, 2018

**Today's Progress** : Temporal Segment Network for action recognition from video

**Thoughts** : Every pieces are engaging with one another, making our action recognition project feasible!

**Link of Works** : https://arxiv.org/abs/1608.00859

## Day 49 : Aug 24, 2018

**Today's Progress** : Recap some of fundamentals in training that network

**Thoughts** : Playing only in higher level applications easily let me to neglect the basics

**Link of Works** : https://youtu.be/hd_KFJ5ktUc

## Day 50 : Aug 25, 2018

**Today's Progress** : Recap some of object detection models 

**Thoughts** : Similar to previous day. I should enhance the basics everytime I get a chance. 

**Link of Works** : https://youtu.be/GxZrEKZfW2o

## Day 51 : Aug 26, 2018

**Today's Progress** : Read some parts of a paper (TSNs: Towards good practices for deep action recognition)

**Thoughts** : I'll repeatedly read the papers on action recognition. Maybe one of the two(TSNs/TS LSTM) could be a baseline for our project.

## Day 52 : Aug 27, 2018

**Today's Progress** : TSNs (Cont.) + watched Activity Recognition videos

**Thoughts** :  
(Things CNNs do well)  
Image classification (solved. ILSVRC err: 2.3%)  
Object localization in an image (maybe solved. ILSVRC err: 6.2%)  
Activity recognition in a video (TS LSTM on UCF101: 94.3%)

**Link of Works** :  
[CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action...](https://youtu.be/pe-_lvrBNzE)  
[CVPR18: Tutorial: Part 1: Human Activity Recognition](https://youtu.be/Flm-kkCqACM)

## Day 53 : Aug 28, 2018

**Today's Progress** : Took some Activity Recognition videos and PyTorch MNIST tutorial

**Thoughts** : Now is the time to match the implementations to the papers. For that purpose I need to get harnessed with PyTorch.

**Link of Works** :  
[Unsupervised Action Localization - ICCV 2017](https://youtu.be/MRN7K_bgerQ)  
[VideoLSTM: Convolves, attends, and flows for action recognition](https://youtu.be/oluw16wExDY)  
[PyTorch in 5 Minutes](https://youtu.be/nbJ-2G2GXL0)  
[Basic MNIST Example](https://github.com/pytorch/examples/tree/master/mnist)

## Day 54 : Aug 29, 2018

**Today's Progress** : Capsule Network, Attention

**Thoughts** : Let´s go Attention

**Link of Works** :  
[Capsule Networks(CapsNets) - Tutorial](https://youtu.be/pPN8d0E3900)  
[Capsule Networks: An Improvement to Convolutional Networks](https://youtu.be/VKoLGnq15RM)  
[Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)

## Day 55 : Aug 30, 2018

**Today's Progress** : Took a Cross Entropy tutorial video

**Thoughts** : Looking back, I've been using cross entropy loss unconsciously, without deep understanding. From the video, Aurélien Géron explains me about cross entropy like I'm 5. I love it.

**Link of Works** : [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://youtu.be/ErfnhcEV1O8)

## Day 56 : Aug 31, 2018

**Today's Progress** : Short exercise on data loading with PyTorch

**Thoughts** : Good. PyTorch gives a useful coding scheme of dealing with datasets.

**Link of Works** : [Data Loading and Processing Tutorial](https://github.com/pytorch/tutorials/blob/master/beginner_source/data_loading_tutorial.py)

## Day 57 : Sep 1, 2018

**Today's Progress** : Exercise on transfer learning with PyTorch

**Thoughts** : I`m getting used to PyTorch style

**Link of Works** : [Transfer Learning tutorial](https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py)

## Day 58 : Sep 2, 2018

**Today's Progress** : PyTorch tutorial (Autograd)

**Thoughts** : Nice feature.. I'm more and more aware that PyTorch is suitable for researchers.

**Link of Works** : [Autograd: automatic differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

## Day 59 : Sep 3, 2018

**Today's Progress** : PyTorch tutorial (Autograd) - revisited

**Thoughts** : Yet, there are some confusing parts of the Autograd to me.

## Day 60 : Sep 4, 2018

**Today's Progress** : PyTorch tutorial (Neural Networks)

**Thoughts** : Cool. I'm getting used to PyTorch.

**Link of Works** : [Neural Networks](file:///home/david/Development/pytorch/tutorials/_build/html/beginner/blitz/neural_networks_tutorial.html)

## Day 61 : Sep 5, 2018

**Today's Progress** : PyTorch tutorial (Training a classifier, ~ 4. Train the network)

**Thoughts** : Easy and comfortable

**Link of Works** : [Training a classifier](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py)

## Day 62 : Sep 6, 2018

**Today's Progress** : PyTorch tutorial (Training a classifier, Data Parallelism)

**Thoughts** : I get a lot of things from PyTorch tutorials not only PyTorch itself but also the way of coding. I think PyTorch way of coding is more Pythonic than other frameworks.

**Link of Works** : (Data Parallelism)[https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html]

## Day 63 : Sep 7, 2018

**Today's Progress** : Learning PyTorch with Examples (~ PyTorch: Defining new autograd functions)

**Thoughts** : Justin Johnson is a great tutor(He is same person with CS231n´s Justin Johnson, right?). I like bottom-up approach, so I'm absorbing his teaching. 

**Link of Works** : (Learning PyTorch with Examples)[https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn]

## Day 64 : Sep 8, 2018

**Today's Progress** : Learning PyTorch with Examples (TensorFlow: Static Graphs ~ PyTorch: Custom nn Modules)

**Thoughts** : Really cool. Anyone landing on PyTorch territory should get this course.

## Day 65 : Sep 9, 2018

**Today's Progress** : Learning PyTorch with Examples (PyTorch: Control Flow + Weight Sharing)

**Thoughts** : Easy and intuitive implementation of vanilla RNN with PyTorch

## Day 66 : Sep 10, 2018

**Today's Progress** : Review - Backpropagation, Cross Entropy

**Thoughts** : Concrete understanding of these concepts are critical since it becomes prevalent of using high level ML frameworks in the fast-moving industries.

**Link of Works** :  
(cs231n:backprop notes)[http://cs231n.github.io/optimization-2/]  
(Softmax classifier)[http://cs231n.github.io/linear-classify/#softmax]  
(Information Theory Basics)[https://justread.link/kGSq-yF5C]  
(A Short Introduction to Entropy, Cross-Entropy and KL-Divergence)[https://youtu.be/ErfnhcEV1O8]

## Day 67 : Sep 11, 2018

**Today's Progress** : Review - PyTorch tutorial (Transfer Learning)

**Thoughts** : Nice to get harnessed with PyTorch helper functions

## Day 68 : Sep 12, 2018

**Today's Progress** : Tried to train Two-Stream ConvNet(PyTorch impl) on custom dataset(UCF101 small set).

**Thoughts** : It's not easy to make custom small set of UCF101. Furthermore, it's not easy to feed custom dataset without problem. Someone said that data management, from gathering to engineering, takes more than 80% of data science tasks, and yes it seems very true!

## Day 69 : Sep 13, 2018

**Today's Progress** : Turned to more hopeful architecture, Hidden Two-Stream Convolutional Networks

**Thoughts** : Just researched some. Interesting idea.

**Link of Works** : (Hidden Two-Stream Convolutional Networks for Action Recognition)[http://arxiv.org/abs/1704.00389v3]

## Day 70 : Sep 14, 2018

**Today's Progress** : Caffe setup for Hidden Two-Stream Convolutional Networks implementation

**Thoughts** : Not easy to install. But I could get over by googling. I think tensorflow<1.0 build from source was way terrible than this.

**Link of Works** : (Hidden-Two-Stream)[https://github.com/bryanyzhu/Hidden-Two-Stream]
