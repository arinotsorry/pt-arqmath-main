## Authors:
Meghan Cantlin, Kiersten Reina, Ari Wisenburn

/*
in the main pt-arqmath-main directory, run:
`chmod +x run_me.sh`
`./run_me.sh`
*/

## Explanation:
This program assumes the BM25 initial ranking has been completed. After the
`make`, `make data`, `make math`, and `make posts` steps are completed
successfully, in the main pt-arqmath-main directory, run:
`chmod +x run_me.sh`
`./run_me.sh`
This script will use a pretrained ColBERT model trained on MSMARCO. With this
model, it will rerank the baseline BM25 model produced in experiment 1. The 
experiment will rerank at depths of 10, 50, and 100 and compare to the results
from experiment 1.

To change the stemmer used on the initial BM25 data, refer to lines 171-172,
225-229, and 346-349 in the
index-arqmath.py file. Comment or uncomment the line for the desired stemmer.
Make sure the stemmer is consistent throughout the file at the points mentioned
above.



