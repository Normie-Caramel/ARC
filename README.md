# ARC
Adaptive Rich-kernelized Contrastive Learning for Capacity Enhancement in Collaborative Filtering

Run the following commands to reproduce the results. The results reported in the paper were generated using random seeds from 2025 to 2034. Please also remember to update the model name in the following two lines in `run.py`.

`config = Config(model=BPR, config_file_list=['config.yaml'])`
`model = BPR(config, train_data.dataset).to(config['device'])`

For SCCF:

`python run.py --model=SCCF --dataset=beauty --learning_rate=10 --train_batch_size=10000 --embedding_size=64 --learner='sgd' --temperature=0.25 --seed=2025`

`python run.py --model=SCCF --dataset=gowalla --learning_rate=10 --train_batch_size=10000 --embedding_size=64 --learner='sgd' --temperature=0.1 --seed=2025`

`python run.py --model=SCCF --dataset=yelp2018 --learning_rate=10 --train_batch_size=10000 --embedding_size=64 --learner='sgd' --temperature=0.2 --seed=2025`

`python run.py --model=SCCF --dataset=pinterest --learning_rate=10 --train_batch_size=10000 --embedding_size=64 --learner='sgd' --temperature=0.1 --seed=2025`

For ARC:

`python run.py --model=MCCF --dataset=beauty --learning_rate=10 --train_batch_size=10000 --embedding_size=64 --learner='sgd' --gain=5.0 --beta=0.5 --seed=2025`

`python run.py --model=MCCF --dataset=gowalla --learning_rate=10 --train_batch_size=10000 --embedding_size=64 --learner='sgd' --gain=10.0 --beta=0.1 --seed=2025`

`python run.py --model=MCCF --dataset=yelp2018 --learning_rate=10 --train_batch_size=10000 --embedding_size=64 --learner='sgd' --gain=5.0 --beta=0.0 --seed=2025`

`python run.py --model=MCCF --dataset=pinterest --learning_rate=10 --train_batch_size=10000 --embedding_size=64 --learner='sgd' --gain=10.0 --beta=0.5 --seed=2025`

For other models:

`python run.py --model={model} --dataset={dataset} --train_batch_size=10000 --embedding_size=64 --seed=2025`