# Deep Learning Approaches to Time Series Analysis

The purpose of this repo is to implement & analyze various deep learning and hybrid approaches to time-series 
classification and regression problems, as well as compare performance to more traditional statistical methods.

## Background

Prior to the 2018 [M4 Time Series Forecasting Competition](https://robjhyndman.com/hyndsight/m4comp/) machine learning sat firmly behind traditional statistical methods when it came to time series analysis in most practitioners' toolkits. This was emphasized by the fact that all submissions which consisted of pure-ML methods performed very poorly (most failing to out-perform even naive methods). Benchmarks available at the [UEA/UCR Time Series Classification repository](http://www.timeseriesclassification.com/) on hundreds of tasks corroborate this view, where ML (and specifically deep-learning) appearances as top algorithm are few and far between.

The [winning submission](https://eng.uber.com/m4-forecasting-competition/) at M4 however, consisted of a hybrid approach that integrated exponential smoothing methods on top of an LSTM network and resulted in a whopping ten percent increase in sMAPE over competition benchmarks. Recent history tells us that when deep learning makes an apperance in a field as a state-of-the-art approach, it tends to become the dominant one shortly thereafter (e.g computer vision and more recently natural language processing). The purpose here is to provide simple implementations of deep learning approaches **that are easy to understand, combine and integrate with each other and classical methods**, notebooks detailing examples of how to use them, and a framework that makes testing and benchmarking on a variety of tasks relatively painless. Feel free to contact me to request implementation of a new idea you have seen or heard about or else add datasets you would like to see some method demonstrated on!

## Deep Learning Approaches

We provide implementations and examples of the following approaches & sub-functionality for both classification and forecasting problems:

1) Recurrent (LSTM)
* optionally add unsupervised pre-training of LSTM classifier with [ULMFiT](https://arxiv.org/abs/1801.06146)-type approach
* optionally add [attentional mechanism](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)

2) [Transformer](https://arxiv.org/abs/1706.03762)-based approaches

3) Convolutional 
* The idea of these approaches is to transform the time series into a 2d-image (or several transformations to yield several images) and apply 2d-convolutional techniques to the image (stack of images).
* Current implementation includes the following image transformations: recurrence plot, gramian angular field, markov transition field, continuous wavelet transform

We emphasize that all these approaches are compatible with one another! We can (for example) combine recurrent and convolutional approaches by concatenating the output of the RNN 'core' with a flattened output of the final convolutional layer.


