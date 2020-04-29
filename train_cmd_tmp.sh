setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name hgt --n_epoch 10 --cuda 0 1> out_cite_hgt.txt 2> log_cite_hgt.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name hetgnn --n_epoch 10 --cuda 1 1> out_cite_hetgnn.txt 2> log_cite_hetgnn.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name han --n_epoch 10 --cuda 2 1> out_cite_han.txt 2> log_cite_han.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name rgcn --n_epoch 10 --cuda 3 1> out_cite_rgcn.txt 2> log_cite_rgcn.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name gat --n_epoch 10 --cuda 0 1> out_cite_gat.txt 2> log_cite_gat.txt

setsid singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_cite.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name gcn --n_epoch 10 --cuda 1 1> out_cite_gcn.txt 2> log_cite_gcn.txt


['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn']