from collections import deque
from .dt_aci import DtACI
import numpy as np

class Human:
    def __init__(self, id, x=15.0, y=15.0):
        self.id = id
        self.past_locations = deque(maxlen=5)
        self.past_timesteps = deque(maxlen=5)
        
        self.past_predictions = deque(maxlen=5)
        self.past_prediction_validity = deque(maxlen=5)

        self.predicted_conformity_scores = []
        self.true_conformity_scores = []

        self._x = x
        self._y = y

        self.last_aci_predicted_conformity_score = np.ones(5)


    def get_position(self):
        return np.array([self._x, self._y])
    
    def store_predictions(self, predictions, is_valid, inlcude_current=True):
        # stack the current position (2, ) in the front of the predictions (5, 2), to form (6, 2)
        if inlcude_current:
            predictions = np.vstack([self.get_position(), predictions])    
        self.past_predictions.append(predictions)
        # print(f"current position: {self.get_position()}")
        # print(f"prediction shape: {predictions.shape}")
        self.past_prediction_validity.append(is_valid)
    
    def set_attributes(self, x, y, t):
        self._x = x
        self._y = y
        self._t = t
        self.past_locations.append(np.array([x, y]))
        self.past_predictions.append(t)

    def reset_aci(self, alpha):
        self.last_prediction = None
        self.last_aci_predicted_conformity_score = np.ones(5)
        self.gt_locations_aci = [] # not conformity score
        self.predictions_aci = [] # not conformity score
        self.pred_error_aci_list = [DtACI(alpha=alpha, initial_pred=(i+1)/10) for i in range(5)] #tag
        # predict 1 step, 2 step, 3 step, 4 step, 5 step conformity score quantile

    def update_aci(self):
        if len(self.gt_locations_aci) == 0 or len(self.predictions_aci) == 0:
            return

        curr_pos = self.gt_locations_aci[-1]
        num_aci_pred_step = min(5, len(self.gt_locations_aci))
        
        for i in range(num_aci_pred_step):
            aci_predictor = self.pred_error_aci_list[i]
            past_ith_prediction = self.predictions_aci[-(i + 1)]  # actually should be past_i+1th_prediction
            # if len(self.gt_locations_aci) >= 2 and i == 0:            #     prediction_start_position = past_ith_prediction[0]
            #     trajectory_past_position = self.gt_locations_aci[-(i + 2)]
            #     past_location_pred_start_dist = np.linalg.norm(prediction_start_position - trajectory_past_position)
            #     print(past_location_pred_start_dist)
            # print(f"past_ith_pred: {past_ith_prediction}")
            predicted_curr_pos = past_ith_prediction[i+1]
            nonconformity_score = np.linalg.norm(curr_pos - predicted_curr_pos)
            # Update the ACI predictor with the ground truth value
            aci_predictor.update_true_value(nonconformity_score)
            # print(nonconformity_score)