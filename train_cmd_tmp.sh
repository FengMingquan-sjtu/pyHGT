====paper-cite-NN====

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name hgt --n_layers 2 --n_epoch 3 --cuda 0 --prior_node_coef 0.8 --prior_relation_coef 0.8 1> out_cite_hgt.txt 2> log_cite_hgt.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name gat --n_epoch 10 --cuda 2 1> out_cite_gat.txt 2> log_cite_gat.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name gcn --n_epoch 10 --cuda 1 1> out_cite_gcn.txt 2> log_cite_gcn.txt


available = ['hgt', 'gcn', 'gat', 'rgcn'];  not available = ['han', 'hetgnn']

====paper-cite-ML====

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _ML --model_dir  /data/fengmingquan/output/HGT --conv_name hgt --n_layers 3 --n_epoch 10 --cuda 0 --prior_node_coef 0.8 --prior_relation_coef 0.8 1> out_cite_hgt.txt 2> log_cite_hgt.txt
setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _ML --model_dir  /data/fengmingquan/output/HGT --conv_name hgt --n_layers 3 --n_epoch 10 --cuda 0 --prior_node_coef 0.0 --prior_relation_coef 0.0 1> out_cite_hgt_2.txt 2> log_cite_hgt.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _ML --model_dir  /data/fengmingquan/output/HGT --conv_name gat --n_epoch 20 --cuda 2 1> out_cite_gat.txt 2> log_cite_gat.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _ML --model_dir  /data/fengmingquan/output/HGT --conv_name gcn --n_epoch 20 --cuda 1 1> out_cite_gcn.txt 2> log_cite_gcn.txt

====paper-cite-CS(may require large mem)====

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _CS --model_dir  /data/fengmingquan/output/HGT --conv_name hgt --n_epoch 100 --cuda 0 1> out_cite_hgt.txt 2> log_cite_hgt.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _CS --model_dir  /data/fengmingquan/output/HGT --conv_name gat --n_epoch 100 --cuda 2 1> out_cite_gat.txt 2> log_cite_gat.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _CS --model_dir  /data/fengmingquan/output/HGT --conv_name gcn --n_epoch 100 --cuda 1 1> out_cite_gcn.txt 2> log_cite_gcn.txt




====paper-author-NN====
setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_author.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name hgt --n_epoch 3 --n_layers 2 --cuda 2 --prior_node_coef 0.8 --prior_relation_coef 0.8 1> out_author_hgt.txt 2> log_author_hgt.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_author.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name hgt --n_epoch 3 --n_layers 3 --cuda 1 --prior_node_coef 0 --prior_relation_coef 0 1> out_author_hgt_2.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_author.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name gat --n_epoch 10 --cuda 0 1> out_author_gat.txt 2> log_author_gat.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_author.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name gcn --n_epoch 10 --cuda 0 1> out_author_gcn.txt 2> log_author_gcn.txt

====paper-author-ML====

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_author.py --data_dir /data/fengmingquan/data --domain _ML --model_dir  /data/fengmingquan/output/HGT --conv_name hgt --n_epoch 10 --cuda 0 --cuda 0 --prior_node_coef 0.8 --prior_relation_coef 0.8 1> out_author_hgt.txt 2> log_author_hgt.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_author.py --data_dir /data/fengmingquan/data --domain _ML --model_dir  /data/fengmingquan/output/HGT --conv_name gat --n_epoch 10 --cuda 0 1> out_author_gat.txt 2> log_author_gat.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_author.py --data_dir /data/fengmingquan/data --domain _ML --model_dir  /data/fengmingquan/output/HGT --conv_name gcn --n_epoch 10 --cuda 0 1> out_author_gcn.txt 2> log_author_gcn.txt






