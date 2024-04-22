# Integrating Graphormer Layers for Enhanced Human-Aware Robot Navigation

## Introduction
This project explores the application of Graphormer, an innovative neural network architecture, for enhancing the navigation capabilities of robots in human-centric environments. Using Graphormer's proficiency with intricate relational data, our goal is to create human-aware robots that can effectively and securely live alongside people.

## Dataset
The research utilizes the SocNav2 dataset (https://github.com/gnns4hri/sngnnv2).

## Methodology
This model's ability to handle graph-structured data efficiently is mostly dependent on Graphormer encodings. To improve the performance of the Transformer architecture in the field of graph representation learning, edge, spatial, and centrality encodings are incorporated. Information about how to use these encodings and why they matter for robot navigation may be found in the README section.

![image](https://github.com/Sungkyung-Shon/sngnn2_Graphormer/assets/81243837/4a246d55-2d0b-43f8-a5f1-931a1da3ac9f)


## Conclusion
The experiments conducted using the SocNav2 dataset illustrate the model's potential in navigating complex social scenarios. While the model's Mean Squared Error (MSE) was higher than that of GAT, MPNN, and R-GCN, it still showed potential for handling complex relational data and generalization to new settings, among other tasks outside MSE evaluation.

## Future Directions
Future goals are to improve Graphormer's interpretability, address its computational issues, and improve its handling of dynamic graphs. Ensuring these systems are capable of moral responsibility and social intelligence during navigation is the aim of this contribution to the field of human-aware robot navigation.
