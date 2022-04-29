# install the pyterrier_colbert package
pip install -q git+https://github.com/terrierteam/pyterrier_colbert.git
# run the pre-trained ColBERT one the 2021 Topics
python3 src/experiment.py ./ARQMath_Collection-post-ptindex ./ARQMath_Evaluation/topics_task_1/2021_topics_task1.xml ./ARQMath_Evaluation/qrels_task_1/2021_qrels_task1.tsv $@

