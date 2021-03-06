# ZstGAN-PyTorch
PyTorch Implementation of "ZstGAN: An Adversarial Approach for Unsupervised Zero-Shot Image-to-Image Translation"
<img src="examples/framework.jpg" />
# Dependency:
Python 3.6

PyTorch 0.4.0

# Usage:
### Unsupervised Zero-Shot Image-to-Image Transaltion
1. Downloading CUB and FLO training and testing dataset following [CUB and FLO](https://pan.baidu.com/s/1m4a4PFpjFNMNLIdE8TlYAQ) with password `n6qd`. Or you can follow the [StackGAN](https://github.com/hanzhanggit/StackGAN) to prepare these two datasets.

2. Unzip the Data.zip and organize the CUB and FLO training and testing sets as:

       Data
       ├── flowers
       |   ├── train
       |   ├── test
       |   └── ...
       ├── birds
           ├── train
           ├── test
           └── ...
    
3. Train ZstGAN on seen domains of FLO:

   `$ python main.py --mode train --model_dir flower --datadir Data/flowers/ --c_dim 102 --batch_size 8 --nz_num 312 --ft_num 2048 --lambda_mut 200`
4. Train ZstGAN on seen domains of CUB:

   `$ python main.py --mode train --model_dir bird --datadir Data/birds/ --c_dim 200 --batch_size 8 --nz_num 312 --ft_num 2048  --lambda_mut 50`
5. Test ZstGAN on unseen domains of FLO at iteration 200000: 

   `$ python main.py --mode test  --model_dir flower --datadir Data/flowers/ --c_dim 102  --test_iters 200000`
6. Test ZstGAN on unseen domains of CUB at iteration 200000: 

   `$ python main.py --mode test  --model_dir bird --datadir Data/birds/ --c_dim 200  --test_iters 200000`
 # Results:
 ### 1. Image translation on unseen domains of FLO at iterations 150000:
 
  **# Results of V-ZstGAN**:
  
  <img src="examples/FLO_v_150000.jpg" width="50%" />
  
  **# Results of A-ZstGAN**:
  
  <img src="examples/FLO_a_150000.jpg" width="50%" />
  
 ### 2. Image translation on unseen domains of CUB at iterations 150000:
 
  **# Results of V-ZstGAN**:
  
  <img src="examples/CUB_v_150000.jpg" width="50%" />
  
  **# Results of A-ZstGAN**:
  
  <img src="examples/CUB_a_150000.jpg" width="50%" />
