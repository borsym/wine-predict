docker build -t wine-quality-model .

docker run wine-quality-model

docker run wine-quality-model --test_size 0.3 --data_path winequality.csv --kernel rbf
