singularity exec --nv -B /data /home/fengmingquan/sandbox/pytorch_geo_1.4 python train_paper_field.py --data_dir /data/fengmingquan/data --domain _NN --model_dir  /data/fengmingquan/output/HGT --conv_name hgt 1> out.txt 2> log.txt