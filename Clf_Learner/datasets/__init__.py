from .csv_dataset import CSVDataset
from .tensor_dataset import TensorDataset

CSV_DATASET_DICT = {
    "ball": "normal_data_dist_5_std_1_5.csv",
    "give_me_some_credit": "credit_scoring_normalised_class_balanced.csv",
    "half_ring": "ball_half_ring_dataset_ball_rad_2_in_rad_3_out_rad_5.csv",
    "half_ring_inverted": "ball_half_ring_inverted_dataset_ball_rad_2_in_rad_3_out_rad_5.csv",
    "ring": "ball_ring_dataset_ball_rad_2_in_rad_3_out_rad_5.csv",
    "twin_moons": "twin_moons.csv",
}