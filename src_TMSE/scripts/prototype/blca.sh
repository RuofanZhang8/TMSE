#!/bin/bash

gpuid=$1

# Loop through different folds
for k in 0 1 2 3 4;
do
	split_dir="/path/to/data_splits/k=${k}"
	split_names="train"
    dataroots="/path/to/data_source"
	bash "./scripts/prototype/clustering.sh" $gpuid $split_dir $split_names $dataroots

done
