python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --save_activations --use_base_model --behaviors altruistic

python plot_activations.py --layers $(seq 0 31) --model_size "7b" --use_base_model --behaviors altruistic

python prompting_with_steering.py --layers $(seq 0 31) --multipliers -0.1 -0.5 -1 -2 -5 -10 0 0.1 0.5 1 2 5 10 --type ab --behaviors altruistic --use_base_model --model_size "7b"
python prompting_with_steering.py --layers $(seq 0 31) --multipliers -0.1 -0.5 -1 -2 -5 -10 0 0.1 0.5 1 2 5 10 --type ab --behaviors altruistic  --model_size "7b"
python plot_results.py --layers $(seq 0 31) --multipliers -0.1 -0.5 -1 -2 -5 -10 0 0.1 0.5 1 2 5 10 --type ab  --behaviors altruistic --use_base_model --model_size "7b"
python plot_results.py --layers $(seq 0 31) --multipliers -0.1 -0.5 -1 -2 -5 -10 0 0.1 0.5 1 2 5 10 --type ab  --behaviors altruistic --model_size "7b"
for i in $(seq 10 15);
do
    python plot_results.py --layers $i --multipliers -0.1 -0.5 -1 -2 -5 -10 0 0.1 0.5 1 2 5 10 --type ab  --behaviors altruistic --use_base_model --model_size "7b"
done

python plot_results.py --layers $(seq 0 31) --multipliers -0.1 -0.5 -1 -2 -5 -10 0 0.1 0.5 1 2 5 10 --type ab --behaviors altruistic --model_size "7b"
for i in $(seq 10 15);
do
    python plot_results.py --layers $i --multipliers -0.1 -0.5 -1 -2 -5 -10 0 0.1 0.5 1 2 5 10 --type ab --behaviors altruistic  --model_size "7b"
done


TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p ./dump/$TIMESTAMP
mv ./results/altruistic ./dump/$TIMESTAMP/results
mv ./normalized_vectors/altruistic ./dump/$TIMESTAMP/normalized_vectors
mv ./vectors/altruistic ./dump/$TIMESTAMP/vectors
mv ./analysis/altruistic ./dump/$TIMESTAMP/analysis
mv ./activations/altruistic ./dump/$TIMESTAMP/activations


python generate_vectors.py --layers $(seq 12 15) --model_size "13b" --save_activations --behaviors wine \
&& python normalize_vectors.py \
&& python plot_activations.py --layers $(seq 12 15) --model_size "13b" --behaviors wine \
&& python prompting_with_steering.py --layers $(seq 12 15) --multipliers -0.1 -0.5 -1 -2 -5 -10 0 0.1 0.5 1 2 5 10 --type ab --behaviors wine --model_size "13b"


python generate_vectors.py --layers $(seq 10 20) --model_size "7b" --save_activations --behaviors wine \
&& python normalize_vectors.py --behaviors wine \
&& python plot_activations.py --layers $(seq 10 20) --model_size "7b" --behaviors wine \
&& python prompting_with_steering.py --layers $(seq 10 20) --multipliers -0.1 -0.5 -1 -2 -5 -10 0 0.1 0.5 1 2 5 10 --type ab --behaviors wine --model_size "7b"