# FDDG
[EAAI 2024] A federated distillation domain generalization framework for machinery fault diagnosis with data privacy


## Paper

Paper link: [A federated distillation domain generalization framework for machinery fault diagnosis with data privacy](https://www.sciencedirect.com/science/article/pii/S0952197623019498)

## Abstract

Federated learning is an emerging technology that enables multiple clients to cooperatively train an intelligent diagnostic model while preserving data privacy. However, federated diagnostic models still suffer from a performance drop when applied to entirely unseen clients outside the federation in practical deployments. To address this issue, a Federated Distillation Domain Generalization (FDDG) framework is proposed for machinery fault diagnosis. The core idea is to enable individual clients to access multi-client data distributions in a privacy-preserving manner and further explore domain invariance to enhance model generalization. A novel diagnostic knowledge-sharing mechanism is designed based on knowledge distillation, which equips multiple generators to augment fake data during the training of local models. Based on generated data and real data, a low-rank decomposition method is utilized to mine domain invariance, enhancing the model's ability to resist domain shift. Extensive experiments on two rotating machines demonstrate that the proposed FDDG achieves a 3% improvement in accuracy compared to state-of-the-art methods.

##  Proposed Network 


![image](https://github.com/CHAOZHAO-1/FDDG/blob/main/IMG/F1.png)

##  BibTex Citation


If you like our paper or code, please use the following BibTex:

```

@article{zhao2024federated,
  title={A federated distillation domain generalization framework for machinery fault diagnosis with data privacy},
  author={Zhao, Chao and Shen, Weiming},
  journal={Engineering Applications of Artificial Intelligence},
  volume={130},
  pages={107765},
  year={2024},
  publisher={Elsevier}
}

```
