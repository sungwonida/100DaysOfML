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
