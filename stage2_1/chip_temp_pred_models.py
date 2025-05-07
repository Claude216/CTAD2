from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import random
import joblib
import torch.nn.functional as F  # Add this import
from sklearn.discriminant_analysis import StandardScaler
import numpy as np
random.seed(42)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================== out chip temp pred model ==============================
"""
    out chip temp pred model: 
        input: 
            XD1, YD1, XD2, YD2, Convection_Film_Coefficient, Internal_Heat_Generation_Magnitude
        output: 
            A, k   (the coefficient of the exponential decay curve)
"""

class OutChipCurvePredictingModel(nn.Module):
    def __init__(self, input_dim, output_dim): 
        super(OutChipCurvePredictingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x): 
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # Output layer without activation for regression
        return x
    



# load the model
# load the weights and initialize the temperature exponential decay prediction model
outchip_curve_pred_model_weight_file_path = "../model_files/12_05_out_chip_temp_decay/curve_pred_model.pth"
outchip_curve_pred_scaler_x_file_path = "../model_files/12_05_out_chip_temp_decay/curve_pred_scaler_x.pkl"
outchip_curve_pred_scaler_y_file_path = "../model_files/12_05_out_chip_temp_decay/curve_pred_scaler_y.pkl"
outchip_curve_pred_scaler_x = MinMaxScaler()
outchip_curve_pred_scaler_y = MinMaxScaler()
outchip_curve_pred_input_dim = 6
outchip_curve_pred_output_dim = 2
outchip_curve_pred_model = OutChipCurvePredictingModel(input_dim=outchip_curve_pred_input_dim, output_dim=outchip_curve_pred_output_dim)
outchip_curve_pred_model.load_state_dict(torch.load(outchip_curve_pred_model_weight_file_path, weights_only=False))
outchip_curve_pred_model.eval()
outchip_curve_pred_scaler_x = joblib.load(outchip_curve_pred_scaler_x_file_path)
outchip_curve_pred_scaler_y = joblib.load(outchip_curve_pred_scaler_y_file_path)


def get_out_chip_decay_curve_coef(chip_len, chip_wid,
                         Convection_Film_Coefficient,
                         Internal_Heat_Generation_Magnitude)->list: 
    

    X_scaled = outchip_curve_pred_scaler_x.transform([[chip_len/2, chip_wid/2, chip_len/2, chip_wid/2, Convection_Film_Coefficient, Internal_Heat_Generation_Magnitude]])

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad(): 
        prediction_scaled = outchip_curve_pred_model(X_tensor)

    prediction_values = outchip_curve_pred_scaler_y.inverse_transform(prediction_scaled.numpy())
    A_pred, k_pred = prediction_values[0][0], prediction_values[0][1]
    return [A_pred, k_pred]

def get_out_chip_temp(A, k, distance_to_edge, T0=35):
    """
        tips: 
            the input data should be like:
                [A, k, distance_to_edge]
    """
    return T0 + get_out_chip_temp_increment(A, k, distance_to_edge)

def get_out_chip_temp_increment(A, k, distance_to_edge):
    
    return A * np.exp(-k * distance_to_edge)


# ============================== in chip temp pred model ==============================

"""
    in chip temp pred model: 
        input: 
            chip_len, chip_wid, Convection_Film_Coefficient, Internal_Heat_Generation_Magnitude, distance_to_center
        output: 
            temperature
"""

class InChipTempPredModel(nn.Module): 
    def __init__(self, input_dim, output_dim=1): 
        super(InChipTempPredModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, output_dim)
    
    def forward(self, x): 
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


inchip_model_weight_file_path = "../model_files/12_13_in_chip/in_chip_pred_model.pth"
inchip_pred_scaler_x_file_path = "../model_files/12_13_in_chip/in_chip_pred_scaler_x.pkl"
inchip_pred_scaler_y_file_path = "../model_files/12_13_in_chip/in_chip_pred_scaler_y.pkl"
inchip_pred_scaler_x = StandardScaler()
inchip_pred_scaler_y = StandardScaler()
inchip_pred_input_dim = 5 
inchip_pred_output_dim = 1
inchip_pred_model = InChipTempPredModel(input_dim=inchip_pred_input_dim,
                                        output_dim=inchip_pred_output_dim)
inchip_pred_model.load_state_dict(torch.load(inchip_model_weight_file_path, map_location=torch.device(device)))
inchip_pred_model.eval()
inchip_pred_scaler_x = joblib.load(inchip_pred_scaler_x_file_path)
inchip_pred_scaler_y = joblib.load(inchip_pred_scaler_y_file_path)

def get_in_chip_temp(chip_len, 
                     chip_wid,
                     Convection_Film_Coefficient, 
                     Internal_Heat_Generation_Magnitude, 
                     distance_to_center )->list:
    
    """
        tips: 
            the input data should be like:
                [chip_len, chip_wid, Convection_Film_Coefficient, Internal_Heat_Generation_Magnitude, distance_to_center]
    """

    X_scaled = inchip_pred_scaler_x.transform([[chip_len, chip_wid, Convection_Film_Coefficient, Internal_Heat_Generation_Magnitude, distance_to_center]])

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad(): 
        prediction_scaled = inchip_pred_model(X_tensor)

    prediction_value = inchip_pred_scaler_y.inverse_transform(prediction_scaled.numpy())
    t_pred = prediction_value[0][0]

    return t_pred 


######################################## Random Forest In chip temp pred #############################################

# RF model for in chip prediction need to retraining or loading later


RF_IC_model = joblib.load("../model_files/01_20/point_temp_RF_IC_model.pkl")

def get_in_chip_temp_RF_IC(chip_len, chip_wid, Convection_Film_Coefficient, Internal_Heat_Generation_Magnitude, distance_to_center): 
    temp_prediction = RF_IC_model.predict([[chip_len, chip_wid, Convection_Film_Coefficient, Internal_Heat_Generation_Magnitude, distance_to_center]])

    return temp_prediction

def get_in_chip_temp_increment_RF_IC(chip_len, chip_wid, Convection_Film_Coefficient, Internal_Heat_Generation_Magnitude, distance_to_center, ambient_temp=35):
    temp_increment = RF_IC_model.predict([[chip_len, chip_wid, Convection_Film_Coefficient, Internal_Heat_Generation_Magnitude, distance_to_center]]) - ambient_temp

    return temp_increment