#!/bin/bash 

for i in kb sup
	do
		for j in senseval2 senseval3 semeval2007 semeval2013 semeval2015 ALL
			do
				echo $i $j
				java Scorer ./data/wsd_eval/WSD_Evaluation_Framework/Evaluation_Datasets/$j/$j.gold.key.txt ./data/results/$i.$j.mean.key
			done
	done
