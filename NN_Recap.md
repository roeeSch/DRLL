# Neural Nets Recap:

![1558614699169](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_1.png)

![1558614884182](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_2.png)

bias as input to layer:

![1558614965687](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_3.png)

More dimentions in input layer:

![1558615088093](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_4.png)

More outputs in the output layer (multiclass):

![1558615184900](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_5.png)

More layers:

![1558615225798](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_6.png)



Feedforward is the process neural networks use to turn the input into an output.

![1558615546771](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_7.png)



The Error Function

![1558615653889](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_8.png)



### Backpropagation

- Doing a feedforward operation.

- Comparing the output of the model with the desired output.

- Calculating the error.

- Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.

- Use this to update the weights, and get a better model.

- Continue this until we have a model that is good.



![1558616047414](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_9.png)



Chain rule:

![1558616120024](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_10.png)

feed forward reformulation:

![1558616307660](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_11.png)

 Error as function of weight, using chain rule on error of feedforward:

![1558616491923](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_12.png)

![img](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_13.png)

So for example:

![1558616641678](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_14.png)

### Early Stopping:

![1558616817851](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_15.png)



Early stopping is stopping when the test set starts to increase.



### Regularization:

![img](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_16.png)

The answer is model 2. This is because:

![1558617064850](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_17.png)

![1558617108358](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_18.png)

The model in the right is too certain - gives little room in applying gradient descent.

**solution:** punish large weights:

Two ways:

![1558617258331](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_19.png)

Trade off betweenregularizations:

![1558617345001](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_20.png) 

In L1 weights tend to go to zero (helps realize which features are informative).

explanation:

![1558617476802](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_21.png)

## Dropout

Sometimes a network uses part of the features which give prominent results neglecting other training  possibilities (other regions of the network).

Solution: During training turn off a different part of the network every few training periods (each epoch disable a differen node in a specific layer).



## Local Minima

Gradient descent is susceptible to converging to local minimas.



## Vanishing Gradient 

the slope of $\sigma$ gets flat very quickly. Its deriviative is close to zero. Since gradient of error with respect to curtain weight is a multiplication of several $\dot{\sigma}$ then it results in a tiny gradient. This results in tiny step in gradient descent towards the minimum error.

**Solutions - Change the activation function:**

1. . $tanh(x)$ goes from -1 to +1 and therefore has better range of derivative. Great this small difference led to great advancements in NN. :-0
2.  relu - maximum between the input and zero.

Usage example:

![1558618675969](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_22.png)



## Batch vs Stochastic Gradient Descent

Batch is computing the error on all the data calculating the gradient and taking a step in the gradient direction. This is quite a heavy computational load and in practice its better to do stochastic gradient descent. SGD is splitting the data (randomly) into small batches and for each batch estimate the gradient (estimate because its only part of the data) , step towards the estimated gradient and repeat for the rest of the batches. 



## Learning Rate Decay

Usually when the model isn't working, reducing the learning rate is beneficial.

Good learning rates are those that decrease as reaching the minimum. Keras has some options for setting these rates (if steep - long step, if plain - small step).



## Local minimum escape strategies:



### Random Restart

Starting gradient descent from random places.This increases the probability of reaching the global minimum (or at leas a good minimum).

### Momentum 

Weighing the previous gradient steps in order to increase the steps in the local minimum:

![1558619644492](/home/roees/DRL course/typoraImages/NN_recap/nn_recap_23.png)

This seems vague, but the algorithm that use momentum work really well in practice. 