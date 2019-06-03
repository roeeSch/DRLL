# Continuous space RL



## Tile coding 

Discreatisizing example:

![1558156055413](typoraImages\Part1\ContRL_TileCoding)

Each location can be located by the tiles it activates and can be represented by a bit vector (ones for tiles activated and zeros elsewhere).

The state value function computation when using this scheme:

<img src="typoraImages\Part1\ContRL_TileCoding_2.png" style="zoom:50">

## Adaptive Tile Coding

This approach doesn't require manually designing the tiles ahead of time.

![1558156789967](typoraImages\Part1\ContRL_AdaptiveTileCoding_1.png)

**Example for devision criteria:** when we are no longer learning from the data (our value function has stopped changing).

**Workshop**: `Tile_Coding.ipynb` (gym: Acrobot-v1)

## Coarse Coding:

![1558192531792](typoraImages\Part1\ConRL_CoarseCoding_1.png)

Each location on the plane is converted into a binary vector, when index i is '1', then it means that the encoded location is in circle i. This is a sparse representation of the plane. 

![01558192617540](typoraImages\Part1\ConRL_CoarseCoding_2)

A more continuous mapping of the area into a vector:

![1558192741443](typoraImages\Part1\ConRL_CoarseCoding_3.png)

## Function approximations:

We are interested in obtaining a good approximating of the actual value function (or q-function). This sometimes requires adding a parameter w:

![1558193315539](typoraImages\Part1\ConRL_functionApprox_1.png)

![1558193381211](typoraImages\Part1\ConRL_functionApprox_2.png)

This is called linear function approximation.

We obtain $W$ by optimization:

![1558193667652](typoraImages\Part1\ConRL_functionApprox_3.png)

This is the rule that we will follow for each sampled state until the error (between the approximate and true state value function).

In order to do this while Q-learning, we need to approximate the action-value function (q).

![1558193958107](typoraImages\Part1\ConRL_functionApprox_4.png)

But why stop here. Lets estimate the state-action**s** value:

![1558194129085](typoraImages\Part1\ConRL_functionApprox_5.png)

Each column of the W matrix emulates a separate linear function.



## Kernel Functions 

![1558194372609](typoraImages\Part1\ConRL_functionApprox_6.png)

We can still use a linear combination of these non-linear features and therefor use linear function approximation.

This allows the value function to represent non-linear relations between the input state and the output value.

## Non-Linear function approximation

![1558194676873](typoraImages\Part1\ConRL_functionApprox_7.png)

This greatly increases our representational capacity of our approximation. This is also the way neural networks work.

We can use gradient descent to optimize and estimate **w**:

![1558194831204](typoraImages\Part1\ConRL_functionApprox_8.png)

This sets us up for deep-reinforcement learning.

