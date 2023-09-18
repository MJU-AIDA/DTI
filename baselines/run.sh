python baselines/baselinemodel.py -td data/vec/vec_train_origin.csv -vd data/vec/vec_dev_origin.csv -m RandomForest
python baselines/baselinemodel.py -td data/vec/vec_train_origin.csv -vd data/vec/vec_dev_origin.csv -m SVM
python baselines/baselinemodel.py -td data/vec/vec_train_origin.csv -vd data/vec/vec_dev_origin.csv -m XGBoost