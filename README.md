
![PyTorch](https://img.shields.io/badge/PyTorch--orange?logo=pytorch)
[![Docker](https://img.shields.io/badge/Docker-Containerization-blue)](https://www.docker.com/)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-Tracking-orange)](https://wandb.ai/)
[![CI/CD](https://img.shields.io/badge/CI/CD-GitHub_Actions-green)](https://github.com/features/actions)
![Git](https://img.shields.io/badge/Git-Version_Control-red?logo=git)



This project is my efforts to deepen my understanding of different (small) large language models by implenting and training them.   


```bash


# To run the code, simply execute:
python train.py

# To run with a specific configuration, create and add a new .yaml file in the config directory, 
# then include it in the command:
python train.py --config_file new_config.yaml
```




















---
---



### To Implement
- [ ] Implement model initialization for training:
    - init_model_from: 'scratch', 'checkpoint', 'pre_trained'
- [ ] Add best model checkpoint functionality
- [ ] Add more tests to verify the train.py code
- [ ] Implement and Train an encoder-decoder transformer architecture

### To Research
- [ ] Understand the impact of dropping transformer components:
    - [Impact of Dropping Transformer Components](https://arxiv.org/pdf/2406.15786)





