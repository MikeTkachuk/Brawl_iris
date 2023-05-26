Quick guide to setup aws services for this repo


### S3
1) Create a bucket
2) Make an IAM instance role with read/write/delete permissions in this bucket

### EC2
3) Request quota increase for g4dn.xlarge instances (0 by default)
4) Create custom AMI (refer to AMI.md for details)
5) Create keypairs and store them in key store locally
... (some standard steps about network and security)
6) Assign the role created in step 2 to the instance
7) Launch the instance and check ssh connection, conda environments and dependencies
8) `$ conda env list` (brawl_stars_cloud should be on the list)
9) `$ conda activate brawl_stars_cloud`  
   `$ python --version`  
   `3.8.2`  
   `$ python`  
   `>>> import torch; torch.cuda.is_available()`  
   `True`

Update instance id in configs to be used in the runs