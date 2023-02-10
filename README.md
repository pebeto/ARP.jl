# ARP.jl
[![Build Status](https://github.com/pebeto/ARPJulia.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pebeto/ARPJulia.jl/actions/workflows/CI.yml?query=branch%3Amain)
This repository contains the [Flux.jl](https://fluxml.ai/) implementation of the Auto-Rotating Perceptrons (Saromo, Villota, and Villanueva) for dense layers of artificial neural networks.

## What is an Auto-Rotating Perceptron? 
The ARP are a generalization of the perceptron unit that aims to avoid the vanishing gradient problem by making the activation function's input near zero, without altering the inference structure learned by the perceptron.

| Classic perceptron | Auto-Rotating Perceptron | 
| --- | --- |
| <img src="https://www.danielsaromo.xyz/assets/img/neuronas_classic.svg" height="200"> | <img src="https://www.danielsaromo.xyz/assets/img/neuronas_ARP.svg" height="200"> | 

[comment]: <> (render en svg y embed en HTML: https://stackoverflow.com/questions/11256433/how-to-show-math-equations-in-general-githubs-markdownnot-githubs-blog)
[comment]: <> (https://stackoverflow.com/questions/47344571/how-to-draw-checkbox-or-tick-mark-in-github-markdown-table)

Hence, a classic perceptron becomes the particular case of an ARP with `rho=1`.

Information extracted from the [original repository](https://github.com/DanielSaromo/ARP). [Reference paper](https://arxiv.org/pdf/1910.02483.pdf).
