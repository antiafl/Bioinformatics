train_svm:
	@python3 train_svm.py --inputs_file inputs_PCA_3.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path=./models

train_kNN:
	@python3 train_kNN.py --inputs_file inputs_PCA_3.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path=./models

train_rfor:
	@python3 train_rfor.py --inputs_file inputs_PCA_3.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path=./models

read_data:
	@python3 read_data.py --inputs_file data.csv --targets_file labels.csv --pca_n 5 --input_storing inputs.csv --targets_storing targets.csv

plot_values:
	@python3 plot_values.py --inputs_file inputs_PCA_2.csv --targets_file targets.csv --model ./models/inputs_PCA_2_svm.pth --dim 2

plot_results:
	@python3 plot_results.py --title 'Comparación de modelos' --input_dir ./results --opt 1
