# Towards Understanding Normalization in Neural ODEs
Repository to reproduce experiments from the  [paper](https://arxiv.org/abs/2004.09222)
 presented at [ICLR 2020 DeepDiffeq workshop](http://iclr2020deepdiffeq.rice.edu/) 
 
 
 
## Requirements
```
numpy
pytorch
```

## Models
We consider neural ODE based models, which are build from ResNets by replacing ResNet blocks with ODE blocks (only blocks that do not reduce spacial dimentions are replaced).

For example, ODENet4 and ODENet10 architectures have the following form:\
- ODENet4:
```conv -> norm -> activation -> ODE block -> avgpool -> fc```\
- ODENet10:
```conv -> norm -> activation -> ResNet block -> ODE block -> ResNet block -> ODE block -> avgpool ->fc```

## Normalization
In  our  experiments,  we  assume  that  normalizations  for  all  ResNet  blocks  are  the  same,  as  well as for all ODE blocks.  Along with these two normalizations, we vary a normalization technique after the first convolutional layer.

Our framework support both layer outputs and layer weights normalization techniques:\
**Layer output normalizations**:
- ```BN```(Batch Normalization),
- ```LN``` (Layer Normaalization),
- ```NormFree``` (absence of layer output normalization)

**Layer weights normalizations**:
- ```WN```(Weight Normalization),
- ```SpecN``` (Spectral Normaalization),
- ```ParamNormFree``` (absence of layer weights normalization) 

## Solver
To propagate through ODE blocks different fixed-step solvers might be used:
- ```Euler```
- ```RK2```
- ```RK4```\
Solver type as well as the number of solver steps are hyperparameters.



## Train ODENet
1. Create a config file with  model, solver and training hypermarameters\
(see, for example ```./config/odenet4_bn-ln_euler-32.cfg```, which is used to train ODENet4 with BN after the first convolutional layer, LN inside ODE block and Euler solver with 32 steps) 
2. Inside  ```./run_training.sh```  modify pathes for
- config file
- data folder (CIFAR-10 will be loaded automatically)
- save folder (log file and checkpoints will be saved in this folder)

3. Run training:
```bash run_training.sh ```

## Reference
If you found this code useful, we appreciate  if you  cite the following paper
```
@article{gusak2020towards,
  title={Towards Understanding Normalization in Neural ODEs},
  author={Gusak, Julia and Markeeva, Larisa and Daulbaev, Talgat and Katrutsa, Alexandr and Cichocki, Andrzej and Oseledets, Ivan},
  journal={arXiv preprint arXiv:2004.09222},
  year={2020}
}
```
