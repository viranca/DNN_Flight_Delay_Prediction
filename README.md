# FLIGHT DELAY ERROR ANALYSIS FORAMSTERDAM AIRPORT

# SCHIPHOL USING A DEEP NEURAL NETWORK (DNN)

```
Christopher M.D.L. Overtveld
Faculty of Aerospace Engineering
Delft University of Technology
c.m.d.l.overtveld@student.tudelft.nl
```
```
Viranca R.I. Balsingh
Faculty of Aerospace Engineering
Delft University of Technology
v.r.i.balsingh@student.tudelft.nl
```
```
January 5, 2020
```
## ABSTRACT

```
Annually the air transport industry loses billions of dollars due to ill predicted flight
delays. Therefore, predicting flight delays has become an important method to re-
duce airline economical losses. In this research, a model is developed to predict
errors in delay estimations using a deep learning neural network, this is in order to
increase prediction accuracy. Flight data from Schiphol Airport covering six months
in the years 2012/2013 is used to train and evaluate this deep neural network. Apart
from the flight data, local weather data is used. The intent is that the model is scal-
able to any airport provided that enough past data is available. Using a deep neural
network an accuracy between 0.5 and 0.65 is conventional for non automated pro-
cesses; the model in this paper is expected to have an accuracy in the range of 0.
to 0.9. Hereby the model will exceed current delay prediction accuracy. The model is
unique in the sense that it uses prediction data provided by the cockpit in real time.
Knowing the actual arrival and departure times, amongst others, ground personnel
and gate allocation can be handled in a more efficient manner, thus resulting in less
idle resources. In a larger time frame this type of model could also be applied to
other forms of transportation, e.g. trains, hyperloop etc.
```
**Keywords** aircraft delay·deep neural network·machine learning·delay prediction

## 1 Introduction

Over the past couple of years, the number of people travelling by air has increased drastically [ 1 ]. Due to
the increase in aircraft arriving at and departing from airports every day, airports have become busier than
ever and, consequently, the possibility of a flight being delayed is highly present. These delays have a large
economical impact and result in negative passenger experience. The cost of these delays for all flights in
the USA in 2007 was 31.2 billion USD [ 2 ]. In the Netherlands, these delays have their largest impact on
Schiphol Airport, which transported 70 million people in 2016 [ 3 ]. Therefore, it is important to keep flight
delays to a minimum, in order to save money and keep the passengers satisfied.

An airplane can be delayed due to various reasons. To analyze and predict the delay one should distinguish
different types of delay. Mueller and Chatterji decide to divide delays into four main types.

```
Out, Off, On, and In (OOOI) times. Out time refers to the time of pushback (specifically when
the parking brake is released). Off time refers to the takeoff time at which weight is no longer
borne on the landing gear. On time is associated with the touchdown time, and the In time
is related to the moment the parking brake is applied at the gate. (Mueller and Chatterji [ 4 ])
```

### 1 INTRODUCTION JANUARY5, 2020

One may then go into more depth to analyze the delays in a more precise manner. The authors [ 4 ] divide
this further into: weather, runway delays, ATC equipment, traffic, and other causes. Truong, Friend and Chen
[ 5 ] use terms as Block Delay and Taxi-out Delay. This paper will use the latter as weather and runway delay
data are not provided, as will be seen in section 3.

Flight phases must also be defined to determine the type of delay. Several studies have been conducted
on the distributions of the different flight phases when predicting flight delays. These phases are departure,
in-flight and arrival. From these studies it is concluded that the departure phase has a Poisson distribution
while the other two phases exhibit a Normal distribution [ 4 ]. Consequently, these distributions are used in the
analysis of flight delay errors.

One method of reducing flight delays is to predict the error in initial delay estimates. Although each airline
makes their own delay predictions, these are not always accurate [ 4 ]. For this reason, making a model to
predict errors in delays is essential since it will help to improve the models used by airlines, which will also
improve passenger experience. Even when the airline’s prediction is given early in the flight, this model
will be able to predict the error of the arrival/departure time already. This is beneficial for Schiphol Airport,
since Schiphol would be able to better act in advance and coordinate gate availability and maintenance.
Passengers will benefit by having more accurate information about their flights at hand and in time. For these
predictive models, previous research has been conducted yet has not led to significant improvements in
predicting flight delays.

For example, Meyn has tried applying probabilistic models [ 6 ]. Mueller and Chatterji have made use of
standard distributions to compute the probability of delays [ 4 ]. Tu, Ball and Jank take yet another approach
using the Expectation-Maximisation algorithm whereas Ding uses a more concise method, namely multiple
linear regression [ 7 , 8 ]. Finally, there are multiple researchers who have tried applying machine learning to
this problem in the past. Based on the aforementioned research, it can be concluded that a neural network
leads to the best accuracy on complex data. Furthermore, it is easy for the user to apply the model after it
has been trained.

A neural network is an application of machine learning which learns to recognise patterns in a data set.
These patterns are used to build a predictive model. The neural network is made up of multiple layers, each
containing weights which are automatically adjusted according to the recognised patterns [9].

For the purpose of discussion, a link to flight operational costs is desired. In order to estimate the overall cost,
A.J. Cook and G. Tanner propose subdividing the problem in the following categories: strategic delays (these
are accounted for in advance), tactical delays (delays occurring in an unexpected manner), and reactionary
delays (this includes costs of delay propagation and other secondary cost factors). Furthermore, their method
approaches the problem by again subdividing the costs in the following categories: Fuel, Maintenance, Fleet,
Crew and passenger. All in all they estimated the total cost of air traffic flow management delay within Europe
to be 1.25 billion Euros (in 2010) [ 10 ]. Moreover, E.B. Peterson et al. performed a similar cost estimation in
the United Stated. Their method comprised of a mathematical model using reference values for fuel and
maintenance cost etc. to transform delay hours into delay costs. Subsequently, by creating a simulation they
estimated a 10 percent decrease in delay time within the US would benefit the US welfare by 17.6 billion US
dollars and a 30 percent decrease in delay time by 38.5 billion US Dollars [11].

This knowledge gap leads to the following research question: how can flight delay estimates be used to
improve customer flight experience at Schiphol Airport? The goal of this research is to create a neural
network model to predict and analyse flight delay errors for Schiphol Airport, with an accuracy (R^2 ) in the
range of 0.50 to 0.65. This is an expected range for non-automated processes [12].

The following structure is used for this paper. Firstly, this paper explains the data handling methods. Secondly,
in the method section, the working principle of neural networks, the method of training, the performance
criteria and optimisation methods are discussed. Thirdly, the results are shown in the subsequent section.
Fourthly, the execution of the method and the results is reflected upon in the discussion. Finally, a conclusion
is drawn, and future recommendations are given.


### 2 LITERATURE REVIEW JANUARY5, 2020

## 2 Literature review

In this section a number of models are mentioned that currently enable the ability to predict these delays and
were primary candidates for this problem. Additionally, current state of machine learning prediction is briefly
discussed.

**2.1 Scenario trees**

En-route from one airport to another, numerous scenarios could occur which cause a flight to be delayed. A
scenario tree is a useful tool to estimate the delay for these scenarios. This tool uses statistical data in order
to assign flight delay to each specific scenario profile. When the pilot of the incoming aircraft updates the
scenario the plane is enduring during flight to the airport, the scenario tree is able to estimate the new arrival
time.

In order to use a scenario tree, it is required to determine all scenario profiles first. Scenario profiles are
constructed based on flight delay data from an airport and are grouped into clusters of similar scenario
profiles, which are interconnected, resulting in a tree structure. Gulpinar et al. have described three different
ways to identify the distinct scenarios: the Simulation and randomised clustering approach, the Optimisation
approach and the Hybrid approach [ 13 ]. Even though the paper by Gulpinar et al. is written for economical
use, the methods for the construction of scenario trees do also apply for the air travel industry. In their work
Liu et al. state that the only efficient way to determine scenario profiles is by the use of an algorithm, for
example a clustering algorithm [14].

When all scenario profiles are clustered, the data is analyzed per cluster. This analysis uses standard
distributions in order to find a delay estimation in case a scenario occurs. The number of clusters in the
scenario tree is of importance; too few clusters results in inaccurate delay predictions, whereas too many
clusters leave an insufficient amount of data per cluster to perform a reliable standard distribution analysis
[13].

Scenario trees have been used in air travel delay predictions previously. In their paper A. Mukherjee and
M. Hansen make use of a scenario tree in order to optimise the single airport ground holding problem, or
SAGHP, in order to minimize delay cost [ 15 ]. Liu et al. use scenario trees to develop a tool for the arrival
capacity of US airports [ 14 ]. In this paper Liu et al. compare two different scenario tree models: Ball et al.
static model [ 16 ] and Mukherjee-Hansen dynamic model which is mentioned above. Lui et al. conclude that
the dynamic model results in lower delay cost opposed to the static model [14].

Scenario tree models can be divided in two groups. A static model, such as used by Ball et al., uses a scenario
tree which remains the same throughout a certain period of time. In order to get a new delay estimation the
static scenario tree has to be run from the start again. A dynamic model, such as the Mukherjee-Hansen
model, anticipates on data that becomes available after a branching point, hence adapting the scenario
profiles based on this data [ 14 ]. In the ideal case, dynamic models result in higher accuracy’s compared to
static models, as was the case for the study conducted by Lui et el. since the mission profiles are up to date.
However, in reality capacity dispersion can lead to mistakes in identifying the scenario profile, which lead to
wrong delay estimations. Nonetheless, the dynamic model reduces the cost by approximately one-third in
comparison to the static model. [14].

**2.2 Machine learning**

Ding [ 8 ] used multi-linear regression to decrease errors seen in basic linear regression using more variables
given to create a better overview of the data. This (multi-) linear regression can be used during advanced
algorithms like machine learning to get a better approximation as was done in previous models [ 7 , 17 , 18 , 19 ].

Unlike conventional prediction models, the neural network model is able to iterate upon previously de-
fined conditions to produce a new, more accurate output. These conditions are formed by learning from
past data. These learning algorithms are based on several methods: K-nearest, random forest, ANN,
Bayesian/regression, gradient descent back propagation.

Lu et al. [ 20 ] compared situations to data from the past, the data that fitted best to the situation was then
used to make a prediction on the propagation of delay time. This prediction is based on the K-nearest
methodology and offers fast reply times.


### 3 DATA PREPARATION JANUARY5, 2020

```
Rebollo and Balakrishnan [ 17 ] took another angle at the problem, and linked the delays of surrounding
airports to the delay times using a random forest machine learning methodology. They were able to predict
delays within a horizon of 2 to 24 hours.
Khanmohammadi, Tutun and Kucuk [ 21 ] attempt to develop a new multi-layer artificial neural network (ANN)
model to predict delay of incoming flights to JFK airport and make a comparison with the traditional back-
propagation method. The new model outperforms in both root mean squared error (difference≈ 0. 023 ) and
training time (difference≈ 1. 6 s).
Sridhar et al. [ 22 ] explore the use of neural network models to predict weather-related flight delays compared
to that of linear/multi-linear regression models. The neural network was generalised using Bayesian
Regularisation and validated with cross-validation. The comparison between models was made with respect
to mean absolute error and root mean squared error, and concluded neural network models performing
slightly better. While this study reviews the performance of neural networks, it is important to also research
the generalisation of such models to be applied with multiple input variables that are both stochastic and
dynamic.
This type of model is explored by Xu et al. [ 23 ] using a Bayesian network (BN) analysis. By separating the
flight schedule into a number of flight segments and then defining delay variables for each, a BN model for
each segment could be created. The major fallback of this method is the lack of focus in errors presented
during the discretization of the regression model used to determine explanatory factors in delay.
```
## 3 Data Preparation

```
The model developed in this research is based on data from Flight Information Royal Dutch Airlines (FIRDA).
This is a system which records messages transmitted between airlines and airports. The data received from
FIRDA consists of delay predictions, made by the arriving or departing aircraft. From two periods of three
months, data is obtained from this system and collected in one file [ 24 ]. This raw data is then processed to
provide the format shown in Table 1.
```
```
Table 1: Data table (first five data points)
OrgFltDate SDep BDep SArr BArr Dep Arr MsgTime NewMsg FltNbr NewTime
23NOV2012 06:15 06:21 07:30 07:38 CDG AMS 22NOV12 16:50 C05 UA 1816 -
23NOV2012 06:15 06:21 07:30 07:38 CDG AMS 23NOV12 06:20 CDQWS UA 1816 -
23NOV2012 06:15 06:21 07:30 07:38 CDG AMS 23NOV12 06:22 0822C UA 1816 07:
23NOV2012 06:15 06:21 07:30 07:38 CDG AMS 23NOV12 06:33 0823C UA 1816 07:
23NOV2012 06:15 06:21 07:30 07:38 CDG AMS 23NOV12 06:56 0828E UA 1816 07:
```
The data consists of multiple message lines per flight. Each line is treated by the neural network as a separate
data point. To clarify the variables that are used in the table; ’BDep/BArr’ is the actual departure/arrival time,
as opposed to the scheduled time (’SDep’ and ’SArr’). This means that the difference between the scheduled
and the block time is the delay time. ’Dep’ and ’Arr’ are the departure and arrival airport, respectively. Next
to that, ’MsgTime’ is the date and moment in time when the delay prediction message was sent. Lastly,
’NewTime’ is the new prediction of departure/arrival time.

```
First of all, the data set is cleaned up. A number of these lines miss data for the actual time of departure
and/or the actual time of arrival. For arriving flights data points without arrival time are excluded from the
data set, and vice versa for departing flights. These data points have no value for the neural network, since
they do not provide a reference time from which the neural network can learn.
Then, the variables in each data point are split into either categorical or continuous variables. Categorising
data allows adding non-continuous and non-numerical data, such as ’airline’, to a neural network, by
translating it to discrete numerical data [ 25 ]. Doing this helps the neural network recognise patterns more
easily, allowing variables such as the time difference between message and arrival/departure to be added as
input explicitly [9].
This represents the initial data input for the model. However, it is found that separating the arrival and
departure data yields better accuracy in predictions in error for each; this is explained in further detail in
section 4, subsection 4.3.
```

### 4 METHOD JANUARY5, 2020

## 4 Method

In order to predict the aircraft delay error, a deep neural network is used. This network is built using the data
mentioned in the previous section. This section provides a detailed view on the model, showing how it works
and explaining why this specific type of network was chosen. The optimisation process and performance
criteria are described as well.

**4.1 Network classification**

In this paper, a deep neural network is built in order to predict the difference between the final delay and the
delay estimated by the pilot. It is built using the Keras library, a machine learning toolkit for Python which is
based on the TensorFlow library [26, 27]^1.

A deep neural network is a neural network with more than one hidden layer. Apart from hidden layers, neural
networks also make use of an input and output layer. The input layer takes the standardised feature columns
from the processed data frame, and the output layer gives a predicted error on the expected delay. The
hidden layers, each consisting of a number of nodes, generate the output.

The neural network built in this research is a feed-forward neural network. Feed-forward is a type of neural
network in which the data is not recycled in-between hidden layers, like in a recurrent neural network. A
recurrent neural network has the ability to fulfil more complex tasks, such as speech analysis, using fewer
hidden layers [ 9 ][ 28 ]. This comes at the cost of higher required computational power. Since the given flight
data is not as complex as speech, a feed-forward type neural network will suffice.

**4.2 Network run-through**

In this section, the full neural network is explained. The section is structured in the same way that the data
flows through the neural network. All six steps are handled chronologically and the used functions are
explained. Even though two different networks are built, one for departure and one for arrival, both of them
have the same structure and therefore the explanation below is applicable for both.

Firstly, the neural network reads the data mentioned in the data input section. In order for the data to be used
by the program, the input data has to be standardised first. Standardising the data is a mean to decrease
the magnitudes of the values in the input data by converting the data using a standard distribution. Using
standardised data will improve the neural network performance, since the program will have to handle a
smaller range of numbers [29].

Secondly, the data is split into a training and testing set. The training set contains the data points that are
used in the construction of the model. This data set uses 70 % of the total standardised input data. The
remaining 30 % makes up the testing set, as this is a commonly used division [ 9 ]. The testing set is used
to find the accuracy of the model that is build from the training data. Since the input data is chronological,
the data points are randomly assigned to the sets. This is done in order to prevent the influence of unseen
patterns with respect to time. Having for example all the winter months in either the training or testing set,
may result in a less accurate model.

Thirdly, the model itself is built. Construction of the model requires the setting of multiple parameters. First of
all, the layers have to be set. For each layer, the number of nodes and activation function are determined.
The purpose of an activation function is to determine whether a given node in a hidden layer is activated or
outputted, and what that output value is. This is determined using the input to that node, in a feed-forward
network this input is the sum of all weighted input data with corresponding biases. In simple terms the
activation function can be seen as a way in which nodes communicate with one another. Keras allows for the
use of multiple activation functions, out of which the rectified linear unit activation function, or ReLU, was
selected for the model [ 30 ]. ReLU is currently the leading activation function in the world and is the most
commonly used activation function for deep neural networks [31].

Next, two more parameters need to be set when building the model: the loss function and the optimiser. The
loss function is required to enable the neural network to make accurate predictions. The loss is a measure
that shows the difference between the prediction of the model and the actual results of the training set.

(^1) The Keras library has also been added to core TensorFlow.


### 4 METHOD JANUARY5, 2020

Decreasing the loss of the network will result in more accurate predictions later on. Keras has multiple loss
functions to choose from [ 32 ]. The program mentioned in this paper uses the mean squared error, or MSE,
in order to train the model. The mean error is the difference between the actual error and the predicted error
in arrival or departure time by the model. The square of this is the mean squared error. The mathematical
representation is given in Equation 1, used in a neural network as the loss function.

#### M SE=

#### 1

```
n
```
```
∑n
```
```
i=
```
```
(Yi−Yˆi)^2 (1)
```
For the reduction of the loss of a neural network, an optimisation algorithm is required. Keras has several
options for the optimisation algorithms [ 33 ], of which multiple are tried in order to find the optimiser that fits
the model best. This turns out to be adaptive momentum estimation, or Adam. Adam is a form of gradient
descent, which means that it adapts the model coefficients, used in the hidden layer(s) along a gradient
towards a stage of minimum error. An explanation for how gradient descent optimisation techniques are used
is presented by Ruder [34].

Essentially, gradient descent aims to minimise the error in a linear regression line fit to the data. By adjusting
the weight and bias associated to said linear fit in a number of iterations, the error is decreased [ 9 ]. This is a
commonly used method for machine learning [ 35 ]. The Adam algorithm has been shown to perform better
than any other optimisation algorithm when deep neural networks are considered [36, 37].

In the fourth step, the model is set up in order for it to run the data. In this step, three parameters are set:
the number of epochs, the batch size and the validation split. An epoch is one run through the training data
set. An extra line of code is added in order to stop the model from running if the loss stops decreasing. This
will benefit the running speed, since no unnecessary epochs are run. The batch size is the number of data
points that is taken into the program before updating the weights of the model. For the model used in this
paper, the mini-batch approach is used. This method only uses a small part of the data set which decreases
processing time but is still accurate, compared to using the entire data set for training [35].

Validation sets are taken from the training set during the training of the model in order to measure how the
model performs on new data. It is a method commonly used to quickly measure the over-fitting of the model
and ensure generalisation. It is essential to make sure that no over-fitting occurs when designing the model.
Over-fitting occurs when the model resembles the training set too much. As a result, the accuracy when
testing the training set is extremely high, but the testing set is not modelled accurately at all. Therefore,
the trade-off is between a model that closely resembles the training set, and also gives the most accurate
general result when tested on the validation set [ 38 ]. The validation split used for the training set is 20 % for
this model.

For the fifth step, the model is run, saving the results from each epoch for later examination. The training and
validation loss are plotted after running the model to visualise if over-fitting has occurred. A separate plot is
made showing the validation mean absolute error, or MAE, in order to pick the number of epochs used for
later predictions. The epoch at which the MAE is smallest is used for making the predictions.

Finally, the prediction is made. The model built in the previous steps is used to estimate the error made by
the pilot for the testing data set. After these predictions are made, the real prediction errors are compared to
the estimated errors found by the model in order to determine the coefficient of determination, orR^2 -value.
TheR^2 -value is the most important indicator for accuracy of the neural network since it shows how well the
model fits the data. The mathematical representation is given in Equation 2. TheR^2 -value is always between
0 and 1, where 0 resembles a bad fit and 1 a good fit; and with this the accuracy of the neural network is
evaluated.

#### R^2 = 1−

#### ∑

```
i(Yi−fi)
```
```
2
∑
i(Yi−
Yˆi)^2
```
#### (2)

**4.3 Optimisation**

After the initial model is built, it has to be optimised to achieve a sufficient accuracy. In this optimisation
process many of the aforementioned parameters are altered in an iterative process in order to create the


### 4 METHOD JANUARY5, 2020

optimal flight delay prediction error model for Schiphol airport. These parameters include the number of
layers, the nodes per layer and the batch size.

Apart from the iteration of the parameters mentioned, several methods are used in an attempt to improve the
performance of the neural network. The first of these methods is the use of data splitting. For the neural
network, the data is split into departing and arriving flights. This is done to create two separate neural
networks, one for the departing and one for the arriving flights. Two separate neural networks have a better
expected accuracy than a single neural network, since the data output (error) for departure and arrival differ
greatly. Due to this difference, one single model is not able to predict both arrival and departure as accurately
as two separate models. In practice, splitting the data increases the accuracy of the neural network and
therefore it is decided to keep this change.

The features used by the neural network have a significant impact on the model performance. The features
used by both the arrival and departure model are: the airline, the scheduled departure time, the scheduled
arrival time and the time left until departure/arrival. Apart from these features, the arrival model uses the
actual departure time and the departure model uses the arriving country name. The reason that the departure
model does not use block departure time is that for real life applications the block departure time will be
unknown, since the plane has not yet departed at the time the prediction is made. Finally, the desired output
for both models is the delay estimation error. This is also imported from the data into the neural network, but
no weights will be assigned to this parameter. The delay estimation error provided is used to check the loss
of the model.

Improving a given model not only entails increasing its accuracy, but also ensuring that the model is able to
perform well on new data; this is called regularisation. This is seen by use of a validation set, a subset of the
training data used in the training of the model. When a large amount of data is present it may be sufficient to
simply split a percentage of the training data before the model is run. K-fold cross-validation is a method
most commonly used when there is little data to work with in order to avoid over-fitting of the model. This is
because it partitions the training into a number of smaller data sets called ’folds’, each of these folds is used
as validation once during training so that the validation data changes with each epoch and the results are
averaged to generate one estimation [ 39 ]. As discussed in section 5, there is very little over-fitting seen in
both models, evident by the constant negative gradient of the validation loss, as the loss of the validation set
continues to decrease with an increased number of epochs. Therefore, the use of K-fold cross-validation
does not lead to an improvement of the neural network and it is decided not to use K-fold in the final model.

More direct regularisation techniques have been explored, similar to the use of cross-validation. These
techniques prevent over-fitting to the training set of the model. The difference lies in the manipulation of the
training input itself in each layer of the network, rather than splitting the training data and selecting multiple
validation sets to be used in training. One such manipulation is termed weight regularisation, whereby
large weights are punished by the loss function. Another method used was dropout, which zeros a certain
percentage of the data input to each layer forcing the network to work with a smaller amount of data and
therefore not form strong patterns seen only in that data. Both techniques have been tried, however, with a
model that does not appear to be over-fitting drastically, the drop in accuracy does not outweigh the gain in
regularisation. Therefore, these techniques are disregarded for the final run.

Grid search is a brute force optimisation method that iterates over a specified range of values for the various
hyper-parameters in a given machine learning model. Taking the hyper-parameters, their value range, and
the optimisation target (e.g. R^2 ) as inputs the algorithm will iterate over each possible combination and
provide the model containing the hyper-parameters needed for the highest target; in this case the highestR^2.

**4.4 Model Design**

A powerful technique used in designing models is something called model ensembling, which uses weighted
averages of the predictions from multiple models built and optimised for the same application but holding
different parameters. This works by cancelling out the biases developed by each model and producing a
more accurate final prediction. This method was attempted for this problem, however the computing time is
increased relatively dramatically for a very limited increase in model accuracy (< 0.02), and has therefore
been excluded from the results.

Each model created for the project is then imported to a primary program file where one is able to input a
given data point, while the program defines the path that data must take; this is done to achieve fluidity and
simplicity in the programs implementation. Finally, the overall flowchart of the model is depicted in figure 1.


### 5 RESULTS JANUARY5, 2020

```
Figure 1: Model Flowchart
```
## 5 Results

In this section, a description is given on what results are achieved from the aforementioned method. The
characteristics of both the arriving flights and departing flights model are presented in Table 2. The layers
presented in this table are the hidden layers constructed for the network. The input layers for both models
are structured the same, as the input data shape is unchanged for each model. Therefore, the output also
has the same shape.

```
Table 2: Results
Parameter Arrival Departure
Architecture
Number of layers 3 3
```
```
Number of nodes
```
```
Layer 1: 256
Layer 2: 256
Layer 3: 128
```
```
Layer 1: 128
Layer 2: 128
Layer 3: 256
Batch size 128 128
Optimiser function used Adam Adam
Use of clusters yes (arrival and departure) yes (arrival and departure)
Features used (input shape) 6 6
Performance
R^2 (accuracy) 0.64 0.
M SE(loss) 0.43 0.
```
Table 2 shows a difference only in the number of nodes per layer of the model, aside from this and the data
input the models performed best with identical architectures.

Figure 2a and Figure 2b display the number of epochs with the corresponding losses. Each time one epoch
is added until the validation loss (continuous line) plateaus. Once the line plateaus, a loss minimum is
reached and the amount of epochs is chosen at this minimum.


### 6 DISCUSSION JANUARY5, 2020

```
(a) Arrival flights. (b) Departure flights.
```
```
Figure 2: Loss per epoch for training and validation data sets for arriving/departing flights.
```
It is evident that the validation loss plateaus and separates from the training loss earlier for the arriving flights
than for the departing flights. This is analogous of over-fitting of the model, i.e. the arriving flight model does
not perform as well on new data for a given number of epochs while the departing flight does not excibit this
problem.

The results achieved after running the arrival model show that after 85 epochs, no better prediction can be
made on new, unseen data. Setting this epoch number for the final training of the model yields anR^2 of 0.
and a loss (mean squared error) of 0.43. Similarly Figure 2b shows prediction stops improving after only 24
epochs; yielding anR^2 of 0.96 and loss of just 0.05. Evidently, the departure flight delay error can be far
better predicted when compared to the arriving flight delay error.

## 6 Discussion

This section first evaluates the reliability of the prediction model results. Then it is oriented towards the
usability and scalability of such a model.

**6.1 Evaluation**

Firstly, it can be observed that the accuracy of the departing flights is a lot higher than the accuracy of the
arriving flights. This can be explained by the fact that the time until departure is usually shorter than the time
until arrival. Therefore it is easier to predict at what time the plane departs than to predict at what time the
plane arrives.

Secondly, the departing planes are still on the ground when the prediction is made. In that case, the plane is
not as prone to weather changes or other, random or environmental causes of delay.

Thirdly, the operations done on the ground are more automated. This ensures there is a smaller variation
in the data due to the consistency of machines. If humans would do the same work, the delay could be
dependable on the employee doing the job. Therefore the difference between the departing and arriving
accuracy are expected.

As mentioned before, a reliableR^2 for non-automated processes lies between 0.50 and 0.65. As the arriving
flights are automated to a lesser extent, the value of 0.64 can be considered reliable for the scope of this
research. As opposed to arrivals, departures may be more automated; additionally there are not as many
exterior influences which can decrease accuracy of the results. This could therefore account for the higher
R^2 value of 0.96 which, together with the high performance of the validation data, can be considered as a
reliable result.

Finally, a flaw in the data handling is the fact that flights which take place overnight at the end of a month,
are disregarded due to difficulties in computing time differences. As the number of these flights is very low
compared to the total number of data points, the impact on the model accuracy is assumed negligible.


### 7 CONCLUSION AND FUTURE WORK JANUARY5, 2020

**6.2 Scalability**

Apart from the ability of using this model on any airport, a similar delay prediction approach could also be
used on other forms of transportation. A requirement for this is the availability/possibility of interim condition
updates. This update could be a new arrival estimate as used in this report, or another variable like new
current position. Another requirement is enough available data to train, how much this exactly should be
depends on the total number of variables and the strength of correlations between them.

## 7 Conclusion and Future Work

In this research, a neural network to predict the errors in flight delay is made, as the impact of delay on
society is large. This is done using a data set with flights arriving at and departing from Schiphol Airport in
the period of six months. This data set is cleaned by adding and removing features to increase the accuracy
of the model. The data set is split into a training and testing set and trained using a feed-forward deep neural
network. This first model is then optimised by increasing the number of hidden layers. These optimised
models can predict delay errors with an accuracy (R^2 ) of 0.96 for departing flights and an accuracy of 0.
for arriving flights.

The research goal has been met with a higher than expected accuracy for the departing flights, and an
expected accuracy for arriving flights.

For future applications, a number of changes and additions can be made to the program in order to increase
accuracy. One of them is to include more relevant variables to the data set, for example weather data at
Schiphol Airport. Another means to improve accuracy is to use a larger data set covering a wider time frame.
To make the model predictions more useful in the short term, instantaneous delay predictions could be made,
using the data from FIRDA directly. Also, provided more data is available, the data could be split into more
sub-components than arriving and departing. An example would be a separate set for each season of the
year. Moreover, multiple deep learning models can be ensembled to increase accuracy at the cost of time for
training the models.

After applying these recommendations, the model can be used by Schiphol airport, but also for other airports
by changing the input data and retraining. Adding live data to the model makes the predictions instantaneous,
making it more useful for airports to use operationally. Using the model, aircraft delays can be predicted a lot
more accurately than the predictions that are currently used. This can decrease costs rising from delays for
airports and airlines.

The most optimal way to implement the model in practice is with the use of an app as this can be made
easily accessible to all travellers. This app will be user-friendly, since the user only needs to insert their flight
number in order to get an accurate prediction of the departure/arrival delay. In this way, airline customers will
be able to know their delay in advance with little to no worry in the time changing again. Since the customers
can now act accordingly to the delay, their airline experience is improved drastically with help from machine
learning.

## Acknowledgements

The authors of this article thank Dr.ir. Bruno Lopes dos Santos for his help and guidance. Furthermore we
thank colleagues Robbie Lunenburg and Teun Vleming for their contribution.


### REFERENCES JANUARY5, 2020

## References

```
[1]Frédéric Dobruszkes. High-speed rail and air transport competition in western europe: A supply-oriented
perspective.Transport policy, 18(6):870–879, 2011.
[2]Michael Ball et al. Total delay impact study: a comprehensive assessment of the costs and impacts of
flight delay in the united states.University of California, Berkeley. Institute of Transportation Studies,
2010.
[3] Royal Schiphol Group. Facts and figures 2016.Schiphol Group, Jan 2017.
[4]Eric Mueller and Gano Chatterji. Analysis of aircraft arrival and departure delay characteristics. In
AIAA’s Aircraft Technology, Integration, and Operations (ATIO) 2002 Technical Forum, pages 58–66,
2002.
[5]Mark A. Friend Dothang Truong and Hongyun Chen. Applications of business analytics in predicting
flight. 57.
[6]Larry Meyn. Probabilistic methods for air traffic demand forecasting.AIAA Guidance, Navigation, and
Control Conference and Exhibit, May 2002.
[7]Yufeng Tu, Michael O Ball, and Wolfgang S Jank. Estimating flight departure delay distributions a
statistical approach with long-term trend and short-term pattern.Journal of the American Statistical
Association, 103(481):112–125, 2008.
[8]Yi Ding. Predicting flight delay based on multiple linear regression. InIOP Conference Series: Earth
and Environmental Science, volume 81, page 012198. IOP Publishing, 2017.
[9]Andreas C. Mueller and Sarah Guido.Introduction to Machine Learning with Python. OReilly Media,
2016.
```
[10] A.J. Cook and G. Tanner.European airline delay cost reference values. 2011.

[11]E.B. Peterson, K Neels, N Barczi, and T Graham. The economic cost of airline flight delay.Journal of
Transport Economics and Policy, 47:107–121, Jan 2013.

[12]Jeffrey M. Wooldridge. Introductory Econometrics: A modern approach. South-Western Cengage
Learning, 2009.

[13]Nalan Gulpinar, Berç Rustem, and Reuben Settergren. Simulation and optimization approaches to
scenario tree generation.Journal of Economic Dynamics and Control, 28(7):1291—-1315, 2004.

[14]Pei chen Barry Liu, Mark Hansen, and Avijit Mukherjee. Scenario-based air traffic flow management:
From theory to practice.Transportation Research Part B: Methodological, 42(7):685–702, 2008.

[15]Avijit Mukherjee and Mark Hansen. A dynamic stochastic model for the single airport ground holding
problem.Transportation Science, 41(4):444—-456, 2007.

[16]Michael O. Ball, Robert Hoffman, Amedeo R. Odoni, and Ryan Rifkin. A stochastic integer program
with dual network structure and its application to the ground-holding problem.Operations Research,
51(1):167—-171, 2003.

[17]Juan Jose Rebollo and Hamsa Balakrishnan. Characterization and prediction of air traffic delays.
Transportation Research Part C: Emerging Technologies, 44:231––241, Apr 2014.

[18]J.R. Bertini and M. do Carmo Nicoletti. An iterative boosting-based ensemble for streaming data
classification. 2018.

[19]Oscar R.P. van Schaijk and Hendrikus G. Visser. Robust flight-to-gate assignment using flight presence
probabilities. 2017.

[20]L. Zonglei, W. Jiandong, and Z. Guansheng. A new method to alarm large scale of flights delay based
on machine learning.International Symposium on Knowledge Acquisition and Modeling, 2008.

[21]Sina Khanmohammadi, Salih Tutun, and Yunus Kucuk. A new multilevel input layer artificial neural
network predicting flight delays at jfk airport.Procedia Computer Science, 95:237–244, 2016.

[22]Banavar Sridhar, Yao Wang, Alexander Klein, and Richard Jehlen. Modeling flight delays and cancella-
tions at the national, regional and airport levels in the united states. 2009.

[23]Ning Xu, Kathryn B. Laskey, Chun hung Chen, Shannon C. Williams, and Lance Sherry. Bayesian
network analysis of flight delays. 2007.

[24]Bruno F. Santos. Theme 4 project: Test, analysis and simulation. Project manual, TU Delft, Delft, The
Netherlands, 2018.


### REFERENCES JANUARY5, 2020

[25]TensorFlow. Feature columns.https://www.tensorflow.org/get_started/feature_columns, Apr
2018.

[26] Francois Chollet et al. Keras.https://keras.io, 2015.

[27]Jeff Dean, Fernanda Viegas, and Martin Abadi. Tensorflow. https://www.tensorflow.org/, Mar
2018.

[28]Ha ̧sim Sak, Andrew Senior, and Françoise Beaufays. Long short-term memory recurrent neural network
architectures for large scale acoustic modeling. InFifteenth annual conference of the international
speech communication association, 2014.

[29]M. Shanker, M.Y. Hu, and M.S. Hung. Effect of data standardization on neural network training.Omega,
24, 1996.

[30] Keras-Documentation. Activations.https://keras.io/activations/, 2018.

[31]Sagar Sharma. Activation functions: Neural networks towards data science. https://
towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6, Sep 2017.

[32] Keras-Documentation. Losses.https://keras.io/losses/, 2018.

[33] Keras-Documentation. Optimizers.https://keras.io/optimizers/, 2018.

[34] Sebastian Ruder. An overview of gradient descent optimization algorithms.ArXiv e-prints, 2016.

[35]Jason Brownlee. A gentle introduction to mini-batch gradient descent
and how to configure batch size. https://machinelearningmastery.com/
gentle-introduction-mini-batch-gradient-descent-configure-batch-size/, Jul 2017.

[36]Anish Singh Walia. Types of optimization algorithms used in neural networks and ways to optimize
gradient descent, Jun 2017.

[37]Diederik P. Kingma and Jimmy Lei Ba. Adam: A method for stochastic optimization. In3rd International
Conference for Learning Representations, San Diego, California, 2015.

[38] Douglas M. Hawkins. The problem of overfitting.ChemInform, 35(19), Nov 2004.

[39] Chollet Francois.Deep learning with Python. Manning Publications Co., 2018.


