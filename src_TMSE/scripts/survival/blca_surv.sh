#!/bin/bash

gpuid=$1
config=$2

task='BLCA_survival'
target_col='os_survival_days'
split_names='train,val'
data_source='/path/to/data_source'
textemb_path='/path/to/text_embeddings'

for k in 0;
do
	split_dir="/path/to/data_splits/k=${k}"
	feat_name='feature_name'
	tags="feature_name_survival_k=${k}_PANTHER"
	bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names $data_source $feat_name $tags $textemb_path 
done
