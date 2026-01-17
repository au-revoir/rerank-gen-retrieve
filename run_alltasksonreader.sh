#!/usr/bin/env bash


echo "-----------------------------------------------------------------------"
echo "Running deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 3 --data_split test --dataset nq --data_path /data/local/gg676/ACL/retrieved_docs/nq/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering"
echo " "
echo " "
deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 3 --data_split test --dataset nq --data_path /data/local/gg676/ACL/retrieved_docs/nq/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering 

echo "-----------------------------------------------------------------------"
echo "Running deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 3 --data_split test --dataset webq --data_path /data/local/gg676/ACL/retrieved_docs/webq/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering"
echo " "
echo " "
deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 3 --data_split test --dataset webq --data_path /data/local/gg676/ACL/retrieved_docs/webq/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering

echo "-----------------------------------------------------------------------"
echo "Running deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 3 --data_split test --dataset tqa --data_path /data/local/gg676/ACL/retrieved_docs/tqa/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering"
echo " "
echo " "
deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 3 --data_split test --dataset tqa --data_path /data/local/gg676/ACL/retrieved_docs/tqa/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering



echo "-----------------------------------------------------------------------"
echo "Running deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 4 --data_split test --dataset nq --data_path /data/local/gg676/ACL/retrieved_docs/nq/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering"
echo " "
echo " "
deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 4 --data_split test --dataset nq --data_path /data/local/gg676/ACL/retrieved_docs/nq/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering 

echo "-----------------------------------------------------------------------"
echo "Running deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 4 --data_split test --dataset webq --data_path /data/local/gg676/ACL/retrieved_docs/webq/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering"
echo " "
echo " "
deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 4 --data_split test --dataset webq --data_path /data/local/gg676/ACL/retrieved_docs/webq/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering

echo "-----------------------------------------------------------------------"
echo "Running deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 4 --data_split test --dataset tqa --data_path /data/local/gg676/ACL/retrieved_docs/tqa/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering"
echo " "
echo " "
deepspeed --num_gpus 8 reader_retrieval.py --document_type rerankt0pp_concat --concat_top_k 4 --data_split test --dataset tqa --data_path /data/local/gg676/ACL/retrieved_docs/tqa/test/rerankinput_bm25/merged_rerankedDoc_results_test.jsonl --prompt_version question_answering
