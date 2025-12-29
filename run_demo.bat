venv\Scripts\activate

python evaluate_and_visualize.py --xgb-data processed_data/xgb_test.csv --lstm-data processed_data/lstm_test.npz --lstm-ids-path processed_data/lstm_test_ids.csv --model-dir trained_models --results-dir results

python demo_threat.py
