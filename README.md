# &ensp;&ensp;&ensp;&ensp;   Ceng489 Term Project 
# &ensp;&ensp;           Network Intrusion Detection Tool

## The Problem
&ensp;&ensp;    In today's technology, network security becomes much harder to maintain with every new enhancement. Every improvement also creates a new anomaly or an option for an intrusion attack on the networks. Our main purpose is to detect the main anomalies and intrusions like DDoS(Distributed Denial-of-Service), Dos (Denial-of-Service), Fuzzing, OS and Service Detection and Port Scanning on given network traffic. This problem trivially is a multiclass classification problem in which a given network traffic event is assigned as normal or other attack categories. To solve this problem, we are given several options to choose from and then train the selected models to detect any anomalies for a given input. 

## Proposed Approach
&ensp;&ensp;    Our problem can be reduced to a multiclass classification problem where for given network traffic, an event can occur and this event may be a normal network flow, an intrusion or an anomaly. We decided to use three different deep learning approaches to tackle this problem. These approaches are AdaBoost (Adaptive Boosting), LSTM (Long-Short Term Memory) and Deep Autoencoders. These neural network applications can classify the network flow for a given dataset. 

    
