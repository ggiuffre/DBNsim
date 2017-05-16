# DBNsim
## A web app for analysing Deep Belief Networks

DBNsim is a web application for training and analysing __Deep Belief Networks__, a particular kind of artifical neural networks. DBNsim has a Python back end (with Django) and a JavaScript front end (which uses mainly Cytoscape.js and Highcharts).

Deep Belief Networks (DBNs) are a particular _architecture_ of neural nets: they are multi-layered networks where each layer is a Restricted Boltzmann Machine (RBM); in practice, a DBN is a "stack" of RBMs. An RBM is a bipartite undirected graph, typically trained with unsupervised learning.
