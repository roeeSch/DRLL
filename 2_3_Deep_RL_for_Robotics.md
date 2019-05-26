# Deep RL for Robotics


## C/C++ API

To successfully leverage deep learning technology in robots, we need to move to a library format that can integrate with robots and simulators.

In this section, we introduce an API (application programming interface) in C/C++. The API provides an interface to the Python code written with PyTorch, but the wrappers use Python’s low-level C to pass memory objects between the user’s application and Torch without extra copies.



## Installing the Repository

------

We will provide the coding environment for you in a Udacity Workspace, so you do not need to install the API. However, if you'd like to install it on a GPU x86_64 system, you need only follow the build instructions in the [repository](https://github.com/udacity/RoboND-DeepRL-Project).

## API Repository Sample Environments

------

In addition to OpenAI Gym samples, the repository contains the following demos:

- C/C++ 2D Samples
  - Catch (DQN text)
  - Fruit (2D DQN)
- C/C++ 3D Simulation
  - (Robotic) Arm (3D DQN in Gazebo)
  - Rover (3D DQN in Gazebo)

The purpose of building the simple 2D samples is to test and understand the C/C++ API as we move toward the goal of using the API for robotic applications. Each of these samples will use a Deep Q-Network (DQN) agent to solve problems.

## The DQN agent

------

The repo provides a base [`rlAgent`](https://github.com/dusty-nv/jetson-reinforcement/blob/master/c/rlAgent.cpp) base class that can be extended through inheritance to implement agents using various reinforcement learning algorithms. We will focus on the [`dqnAgent`](https://github.com/dusty-nv/jetson-reinforcement/blob/master/c/dqnAgent.cpp) class and applying it to solve DQN reinforcement learning problems.

The following pseudocode illustrates the signature of the [`dqnAgent`](https://github.com/dusty-nv/jetson-reinforcement/blob/master/c/dqnAgent.cpp) class:

```cpp
class dqnAgent : public rlAgent
{
public:

    /**
     * Create a new DQN agent training instance,
     * the dimensions of a 2D image are expected.
     */
    static dqnAgent* Create( uint32_t width, uint32_t height, uint32_t channels, 
        uint32_t numActions, const char* optimizer = "RMSprop", 
        float learning_rate = 0.001, uint32_t replay_mem = 10000, 
        uint32_t batch_size = 64, float gamma = 0.9, float epsilon_start = 0.9,  
        float epsilon_end = 0.05,  float epsilon_decay = 200,
        bool allow_random = true, bool debug_mode = false);

    /**
     * Destructor
     */
    virtual ~dqnAgent();

    /**
     * From the input state, predict the next action (inference)
     * This function isn't used during training, for that see NextReward()
     */
    virtual bool NextAction( Tensor* state, int* action );

    /**
     * Next action with reward (training)
     */
    virtual bool NextReward( float reward, bool end_episode );
}
```

In the pseudocode above, the agent is instantiated by the `Create()` function with the appropriate initial parameters. For each iteration of the algorithm, the environment provides sensor data, or environmental state, to the `NextAction()` call, which returns the agent's action to be applied to the robot or simulation. The environment's reward is issued to the `NextReward()` function, which kicks off the next training iteration that ensures the agent learns over time.

Let's take a detailed look at some of the parameters that can be set up in the `Create()` function.

## Setting the Parameters

------

The parameter options are specified separately for each sample. For instance, you can see how the parameters are set for the `catch` agent by perusing the top of the [`catch.cpp`](https://github.com/dusty-nv/jetson-reinforcement/blob/master/samples/catch/catch.cpp) file.

```
// Define DQN API settings
#define GAME_WIDTH   64             // Set an environment width 
#define GAME_HEIGHT  64             // Set an environment height 
#define NUM_CHANNELS 1              // Set the image channels 
#define OPTIMIZER "RMSprop"         // Set a optimizer 
#define LEARNING_RATE 0.01f         // Set an optimizer learning rate
#define REPLAY_MEMORY 10000         // Set a replay memory
#define BATCH_SIZE 32               // Set a batch size
#define GAMMA 0.9f                  // Set a discount factor
#define EPS_START 0.9f              // Set a starting greedy value
#define EPS_END 0.05f               // Set a ending greedy value
#define EPS_DECAY 200               // Set a greedy decay rate
#define USE_LSTM true               // Add memory (LSTM) to network
#define LSTM_SIZE 256               // Define LSTM size
#define ALLOW_RANDOM true           // Allow RL agent to make random choices
#define DEBUG_DQN false             // Turn on or off DQN debug mode
```



## Fruit Sample

The [`fruit`](https://github.com/dusty-nv/jetson-reinforcement/blob/master/samples/fruit/fruit.cpp) sample is a simple C/C++ program which links to the reinforcement learning library provided in the repository.

The environment is a 2-dimensional screen. The agent is learning "from vision" to translate the raw pixel array into actions using the DQN algorithm. The agent appears at random locations and must find the "fruit" object to gain the reward and win episodes before running out of bounds or the timeout period expires. The agent has 5 possible actions to choose from: up, down, left, right, or none on the screen in order to navigate to the object.

<img src='typoraImages/Part2/RoboticsNvidiaGym_01.gif'>

## `fruit` Implementation

------

The [`fruit`](https://github.com/dusty-nv/jetson-reinforcement/blob/master/samples/fruit/fruit.cpp) code looks very similar to the `catch` code. It’s important to note that the *same agent class* is used in both environments!

```cpp
    // Create reinforcement learner agent in pyTorch
    dqnAgent* agent = dqnAgent::Create(gameWidth, gameHeight, 
                       NUM_CHANNELS, NUM_ACTIONS, OPTIMIZER, 
                       LEARNING_RATE, REPLAY_MEMORY, BATCH_SIZE, 
                       GAMMA, EPS_START, EPS_END, EPS_DECAY,
                       USE_LSTM, LSTM_SIZE, ALLOW_RANDOM, DEBUG_DQN);
```

The parameter values are slightly different (the frame size and number of channels have changed), but the algorithm for training the network to produce actions from inputs remains the same.

The environment is more complicated for `fruit` than it is for `catch`, so it has been extracted to the [`fruitEnv.cpp`](https://github.com/dusty-nv/jetson-reinforcement/blob/master/samples/fruit/fruitEnv.cpp) module and it’s own class, `FruitEnv`. The environment object named `fruit` is instantiated in the `fruit.cpp` module.

```cpp
    // Create Fruit environment
    FruitEnv* fruit = FruitEnv::Create(gameWidth, gameHeight, epMaxFrames);
```

We can trace the handoff between the agent and environment through the following code snippet located in the main game loop in the `fruit.cpp` module:

```cpp
        // Ask the agent for their action
        int action = 0;
        if( !agent->NextAction(input_tensor, &action) )
            printf("[deepRL]  agent->NextAction() failed.\n");

        if( action < 0 || action >= NUM_ACTIONS )
            action = ACTION_NONE;

        // Provide the agent's action to the environment
        const bool end_episode = fruit->Action((AgentAction)action, &reward);
```

In this snippet, `action` is the variable that contains the `agent` object’s next action, based on the previous environment state represented by the `input_tensor` variable. The `reward` is determined in the last line when the `action` is submitted to the environment object named `fruit`.



### Quiz - Fruit Rewards

------

The `fruit` rewards function can be implemented a number of different ways. Below are several possible reward functions for the game that compare previous and current distances between the agent and its goal. Match each to descriptions in the quiz below.



### A

```cpp
*reward = (lastDistanceSq > fruitDistSq) ? 1.0f : 0.0f;
```

### B

```cpp
*reward = (sqrtf(lastDistanceSq) - sqrtf(fruitDistSq)) * 0.5f;
```

### C

```cpp
*reward = (sqrtf(lastDistanceSq) - sqrtf(fruitDistSq)) * 0.33f;
```

### D

```cpp
*reward = exp(-(fruitDistSq/worldWidth/1.5f));
```



Match each description to the reward functions shown above.

A - 1 if closer than previously; 0 otherwise

B - half the value of the change in distance

C - a third of the change in distance

D - exponential value of -1.5 multiplied by the ratio of the distance squared over the screen width



## Running `fruit`

------

To test the fruit sample, open the desktop in the **Udacity Workspace**, open a terminal, and once again navigate to the folder containing the samples with:

```bash
$ cd /home/workspace/jetson-reinforcement/build/x86_64/bin
```

Launch the executable from the terminal:

```bash
$ ./fruit
```

It should achieve 85% accuracy after around ~100 episodes within the default 48x48 environment.

## Alternate Arguments

------

Optional command line parameter examples for `fruit` can be used to change the size of the pixel array and limit the number of frames:

```bash
$ ./fruit --width=64 --height=64 --episode_max_frames=100
```