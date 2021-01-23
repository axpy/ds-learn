# Chapter 1.

## Exercices

1. How would you define Machine Learning?
    Machine learning is a sphere, which provides a set of technologies allowing to solve some specific tasks without explicitly programming the methods of solving them.

2. Can you name four types of problems where it shines?
    1. Recognition problems (too many arguments, no regular solution exists, non-general solution)
    2. Prediction
    3. Searching patterns (data mining)
    4. Quick adaptation for new data

3. What is a labeled training set?
    Supervized training - when each sample initially has an answer - what is it?

4. What are the two most common supervised tasks?
    Prediction and classification

5. Can you name four common unsupervised tasks?
    1. Classification form clusterization
    2. Data mining from association rule learning
    3. Anomaly detection
    4. Visualization

6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?
    Reinforcement learning (allow to perfom actions with penalties and rewards)

7. What type of algorithm would you use to segment your customers into multiple groups?
    Classification by clusterization

8. Would you frame the problem of spam detection as a supervised learning prob‐
lem or an unsupervised learning problem?
    Supervised or semisupervised, as machine should know at first, which emails people usually label as a spam.

9. What is an online learning system?
    System, which doesn't require 're-learn' everything from scratch due to new incoming data. Such system can learn something new on top.

10. What is out-of-core learning?
    Working on data, that is too big to fit on ram. Separating data on batches (or use of online learning) is the key

11. What type of learning algorithm relies on a similarity measure to make predic‐
tions?
    Instance-base learning, when system learns by heart, than applying some similarities it finds is it same enough or not (by similarity rate).

12. What is the difference between a model parameter and a learning algorithm’s
hyperparameter?
    Parameters are constanly changing while machine is learning to improve score. Hyperparameters are set at the beginning of learning process, e.g regularization. Parameters - parameter of model, hyperparameters - parameters of the learning algorithm.

13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?  
    Search for parameters dependencies, trends, by which constructing functions. Then using new instances' parameter in these functions and getting result.

14. Can you name four of the main challenges in Machine Learning?
    Bad data, bad features, overfit, underfit.

15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?
    Overfit problem: increase regularization, decrease features, increase data set

16. What is a test set, and why would you want to use it?
    Test set is a set, that wasn't included while training.

17. What is the purpose of a validation set?
    Validation set is used to find best hyperparameters, or models. Train on train set, validate on validation - repeat these two steps for all possible options (like regularization parameter, model type, etc). Then find the generalized error by applying parameters (theta) on test set.

18. What is the train-dev set, when do you need it, and how do you use it?
    Train-dev set comes alongside with validation set, same size, same place. It's just for additional confidance, that train set isn't overfitted and validation set doesn't differ from the train set very much.

19. What can go wrong if you tune hyperparameters using the test set?
    The model will be more precise learned to test data, and will perform more poorly on the new instances.











































