python3 train_svm.py --inputs_file inputs_PCA_2.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
python3 train_svm.py --inputs_file inputs_PCA_3.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
python3 train_svm.py --inputs_file inputs_PCA_5.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
python3 train_svm.py --inputs_file inputs_PCA_10.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models

python3 train_kNN.py --inputs_file inputs_PCA_2.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
python3 train_kNN.py --inputs_file inputs_PCA_3.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
python3 train_kNN.py --inputs_file inputs_PCA_5.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
python3 train_kNN.py --inputs_file inputs_PCA_10.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models

python3 train_rfor.py --inputs_file inputs_PCA_2.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
python3 train_rfor.py --inputs_file inputs_PCA_3.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
python3 train_rfor.py --inputs_file inputs_PCA_5.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
python3 train_rfor.py --inputs_file inputs_PCA_10.csv --targets_file targets.csv --noftrains 100 --test_size=0.4 --results_path ./results --models_path ./models
