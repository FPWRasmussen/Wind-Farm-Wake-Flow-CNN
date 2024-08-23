import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from py_wake.deficit_models import NOJDeficit
from py_wake.deficit_models.deficit_model import (BlockageDeficitModel,
                                                  WakeDeficitModel)
from py_wake.deficit_models.gaussian import (BastankhahGaussianDeficit,
                                             TurboGaussianDeficit)
from py_wake.flow_map import HorizontalGrid
from py_wake.site._site import UniformSite
from py_wake.superposition_models import SquaredSum, SuperpositionModel
from py_wake.turbulence_models import TurbulenceModel
from py_wake.utils.model_utils import get_models
from py_wake.wind_farm_models import All2AllIterative, PropagateDownwind
from py_wake.wind_farm_models.wind_farm_model import SimulationResult
from shapely.geometry import Point, Polygon
from generate_pywake_sim.SimpleTurbine import SimpleTurbine


class ConfigLoader:
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        self._config = config_dict

    def __getattr__(self, attr):
        value = self._config.get(attr, {})
        if isinstance(value, dict):
            return ConfigLoader(value)
        else:
            return value

    def __repr__(self):
        return repr(self._config)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(config_dict)



def load_engineering_models(config):
    deficitModels = {
        model.__name__: model() 
        for model in get_models(WakeDeficitModel)
    }

    superpositionModels = {
        model.__name__: model() 
        for model in get_models(SuperpositionModel)
    }

    # blockageDeficitModels = {
    #     (model.__name__ if model is not None else "None"): model() 
    #     for model in get_models(BlockageDeficitModel)
    # }
    blockageDeficitModels = {
    (model.__name__ if model is not None else "None"): (model() if model is not None else None)
    for model in get_models(BlockageDeficitModel)
}

    turbulenceModels = {
    (model.__name__ if model is not None else "None"): (model() if model is not None else None)
    for model in get_models(TurbulenceModel)
}
    # turbulenceModels = {
    #     (model.__name__ if model is not None else "None"): model() 
    #     for model in get_models(TurbulenceModel)
    # }

    return (deficitModels[config.wake_deficit_model], 
            superpositionModels[config.superposition_model],
            blockageDeficitModels[config.blockage_deficit_model],
            turbulenceModels[config.turbulence_model])

def calculate_isoperimetric_quotient(geometry):
    perimeter = geometry.exterior.length
    area = geometry.area
    radius_of_circle = perimeter / (2 * np.pi)
    area_of_circle = np.pi * radius_of_circle**2
    isoperimetric_quotient = area / area_of_circle
    return isoperimetric_quotient
    

def generate_conditions(config):
    inflow = config.inflow_settings
    geo = config.wind_farm_geometry_settings
    order = 3
    data = {"wind_speed": np.random.uniform(inflow.V_min, inflow.V_max),
            "turbulence_intensity": np.random.uniform(inflow.TI_min, inflow.TI_max),
            "wind_direction": np.random.uniform(inflow.WD_min, inflow.WD_max),
            "turbine_spacing": np.random.uniform(geo.turbine_spacing_min, geo.turbine_spacing_max),
            "turbine_position_noise": np.random.uniform(geo.turbine_position_noise_min, geo.turbine_position_noise_max),
            "polygon_avg_radii": np.random.uniform(geo.polygon_avg_radii_min**order, geo.polygon_avg_radii_max**order)**(1/order),
            "polygon_irregularities": np.random.uniform(geo.polygon_irregularities_min, geo.polygon_irregularities_max),
            "polygon_spikinesses": np.random.uniform(geo.polygon_spikinessess_min, geo.polygon_spikinessess_max),
            "polygon_num_vertices": np.random.randint(geo.polygon_num_vertices_min, geo.polygon_num_vertices_max + 1)
            }

    return pd.DataFrame([data])

def generate_polygon(stats : pd.Series) -> Polygon:
    def random_angle_steps(steps: int, irregularity: float) -> np.ndarray:
        """Generates the division of a circumference in random angles.

        Args:
            steps (int):
                the number of angles to generate.
            irregularity (float):
                variance of the spacing of the angles between consecutive vertices.
        Returns:
            np.ndarray: the array of the random angles.
        """
        # generate n angle steps
        lower = (2 * np.pi / steps) - irregularity
        upper = (2 * np.pi / steps) + irregularity
        angles = np.random.uniform(lower, upper, steps)
        angle_sum = np.sum(angles)

        # normalize the steps so that point 0 and point n+1 are the same
        angles *= ((2*np.pi)/angle_sum)
        return angles

    def clip(value, lower, upper):
        """
        Given an interval, values outside the interval are clipped to the interval edges.
        """
        return np.minimum(upper, np.maximum(value, lower))
    
    irregularity = stats.polygon_irregularities.copy()
    spikiness = stats.polygon_spikinesses.copy()
    irregularity *= 2 * np.pi / stats.polygon_num_vertices
    spikiness *= stats.polygon_avg_radii
    angle_steps = random_angle_steps(stats.polygon_num_vertices, irregularity)

    angles = np.cumsum(angle_steps)
    radii = clip(np.random.normal(stats.polygon_avg_radii, spikiness, stats.polygon_num_vertices), 0.2*stats.polygon_avg_radii.item(), 5*stats.polygon_avg_radii.item()) # maybe look here
    points = np.column_stack((radii * np.cos(angles),
                              radii * np.sin(angles)))
    return Polygon(points)

def rotate_points_around_centroid(points, angle_degrees, center_of_rotation=None, random_direction: bool = True):
   """
   Rotate a set of 2D points around a specified centroid by a given angle.

   Args:
       points (numpy.ndarray): An array of shape (n, 2) or (2, n) representing the (x, y) coordinates of the points.
       angle_degrees (float): The angle to rotate the points by, in degrees.
       center_of_rotation (list or numpy.ndarray, optional): The (x, y) coordinates of the centroid to rotate around.
           If None (default), the centroid is calculated as the mean of the points.
       random_direction (bool, optional): If True (default), the angle is randomly chosen between -angle_degrees and angle_degrees.

   Returns:
       numpy.ndarray: An array of shape (n, 2) representing the rotated points.

   Example:
       >>> points = np.array([[1, 2], [3, 4], [5, 6]])
       >>> rotated_points = rotate_points_around_centroid(points, 90)
       >>> print(rotated_points)
       [[-2.  1.]
        [-4.  3.]
        [-6.  5.]]
   """
   points = np.array(points)

   # Ensure the points have the correct shape (n, 2)
   if points.shape[0] != 2:
       points = points.T

   angle_radians = np.deg2rad(angle_degrees)  # Convert the angle to radians

   if random_direction:
       angle_radians = np.random.uniform(low=-angle_radians, high=angle_radians)

   # Calculate the centroid if not provided
   if center_of_rotation is None:
       center_of_rotation = np.mean(points, axis=1, keepdims=True)
   else:
       center_of_rotation = np.array(center_of_rotation).reshape(2, 1)

   centered_points = points - center_of_rotation  # Subtract the centroid from each point

   # Create the rotation matrix
   rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                               [np.sin(angle_radians), np.cos(angle_radians)]])

   rotated_centered_points = rotation_matrix @ centered_points  # Apply the rotation matrix
   rotated_points = rotated_centered_points + center_of_rotation  # Add back the centroid

   return rotated_points

def generate_windpark(polygon : Polygon, stats : pd.Series) -> np.ndarray:
    """
    Generate noisy grid points within the bounding box of a polygon.

    Args:
        polygon: Polygon object representing the area of interest.
        spacing (float): Spacing between grid points.

    Returns:
        np.ndarray: Array containing noisy grid points
               inside the specified polygon.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    x_points = np.arange(min_x, max_x, stats.turbine_spacing.item())
    y_points = np.arange(min_y, max_y, stats.turbine_spacing.item())
    X, Y = np.meshgrid(x_points, y_points)
    coordinates = np.column_stack((X.ravel(), Y.ravel()))
    point_mask = np.zeros(len(coordinates), dtype=bool)

    for i, coord in enumerate(coordinates):
        if polygon.contains(Point(coord)):
            point_mask[i] = True

    point_mask = point_mask.reshape(X.shape)
    X_masked, Y_masked = X[point_mask], Y[point_mask]
    X_rotated, Y_rotated = rotate_points_around_centroid(np.array([X_masked, Y_masked]), 45, center_of_rotation=None, random_direction = True)
    X_noisy = X_rotated + np.random.normal(loc=0, scale=stats.turbine_spacing * stats.turbine_position_noise, size=X_masked.shape)
    Y_noisy = Y_rotated + np.random.normal(loc=0, scale=stats.turbine_spacing * stats.turbine_position_noise, size=Y_masked.shape)

    if X_noisy.size > 0:
        X_noisy = X_noisy - np.max(X_noisy) # Move the wind park to negative coordinates
    else:
        return np.vstack([X_noisy, Y_noisy])
    
    return np.vstack([X_noisy, Y_noisy])



def init_pywake_sim(config, wind_park, stats, wind_turbine):
    site = UniformSite()
    
    wake_deficitModel, superpositionModel, blockage_deficitModel, turbulenceModel = load_engineering_models(config.flow_model_settings)
        
    if config.flow_model_settings.propagation_method == "All2AllIterative":
        wf_model = All2AllIterative(site, wind_turbine,
                                    wake_deficitModel=wake_deficitModel,
                                    superpositionModel=superpositionModel,
                                    blockage_deficitModel=blockage_deficitModel,
                                    turbulenceModel=turbulenceModel)
        
    elif config.flow_model_settings.propagation_method == "PropagateDownwind":
        wf_model = PropagateDownwind(site, wind_turbine,
                            wake_deficitModel=wake_deficitModel,
                            superpositionModel=superpositionModel,
                            turbulenceModel=turbulenceModel)
    else:
        raise ValueError("Invalid propagation method: Use 'All2AllIterative' or 'PropagateDownwind'")
    
        
    sim_res = wf_model(*wind_park,                      # Wind turbine positions
                        wd = stats.wind_direction,                  # Wind direction
                        ws= stats.wind_speed,                   # Wind speed 
                        TI= stats.turbulence_intensity)                   # Turbulence intensity    

    return sim_res

def check_valid_config(config):
    turbine_amount_max = config.wind_farm_geometry_settings.turbine_amount_max
    turbine_spacing_min = config.wind_farm_geometry_settings.turbine_spacing_min
    polygon_avg_radii_max = config.wind_farm_geometry_settings.polygon_avg_radii_max

    # max_expected_turbines = (np.pi * polygon_avg_radii_max * 2) / (turbine_spacing_min ** 2)

    # if max_expected_turbines > turbine_amount_max:
    #     user_prompt = input("The current settings for 'turbine_spacing_min' ({}) and 'polygon_avg_radii_max' ({}) can produce wind farms containing more than {:.0f} turbines. The program is limited to outputting farms with {} turbines. Are you sure you want to continue? [y/N] ".format(turbine_spacing_min, polygon_avg_radii_max, max_expected_turbines, turbine_amount_max))

    #     if user_prompt.lower() == "y":
    #         print("Continuing simulations...")
    #     else:
    #         raise KeyboardInterrupt("Terminating the program...")
    
    if any(x < 0 or x > 1 for x in [config.wind_farm_geometry_settings.polygon_irregularities_min,
                                    config.wind_farm_geometry_settings.polygon_irregularities_max,
                                    config.wind_farm_geometry_settings.polygon_spikinessess_min,
                                    config.wind_farm_geometry_settings.polygon_spikinessess_min]):
        raise ValueError("Irregularity and Spikiness must be between 0 and 1.")

def load_sim_res(file_path):
    site = UniformSite()
    wf_model = All2AllIterative(site, SimpleTurbine(),
                                wake_deficitModel=TurboGaussianDeficit(),
                                superpositionModel=SquaredSum(),
                                blockage_deficitModel=None,
                                turbulenceModel=None)

    sim_res = SimulationResult.load(file_path, wf_model)
    return sim_res

 
def check_valid_wind_farm_layout(config, wind_park):
    # check turbine amount
    if config.wind_farm_geometry_settings.turbine_amount_min > wind_park.shape[1] > config.wind_farm_geometry_settings.turbine_amount_max:
        return False
    
    # check x coordinates
    if (wind_park[0, :] < config.pywake_results_settings.x_range_min).any():
        return False
    
    # check y coordinates
    if ((wind_park[1, :] < config.pywake_results_settings.y_range_min) | (wind_park[1, :] > config.pywake_results_settings.y_range_max)).any():
        return False
    
    return True

def run(config):
    # stats_folder = os.path.join(config.pywake_results_settings.dataset_path, "stats")
    stats_geo = pd.DataFrame()
    stats_sim = pd.DataFrame()
    temp_wind_park_list = []
    if not config.pywake_results_settings.load_pregenerated_farms:
        for _ in range(config.pywake_results_settings.num_simulations):
            for _ in range(config.wind_farm_geometry_settings.max_generation_attempts):
                data = generate_conditions(config)
                polygon = generate_polygon(data)
                wind_park = generate_windpark(polygon, data)
                
                valid_layout = check_valid_wind_farm_layout(config, wind_park)
                
                if valid_layout:  # Continue with the script if valid_layout is True
                    break
            else: 
                raise RuntimeError("Failed to generate a valid wind farm layout after max attempts")
            
            temp_wind_park_list.append(wind_park)
            stats_layout = pd.DataFrame(data={
            "wind_turbine_amount": [len(wind_park[1])],
            "polygon_area": [polygon.area],
            "polygon_roundness": [calculate_isoperimetric_quotient(polygon)]})

            data = pd.concat([data, stats_layout], axis=1)
            stats_geo = pd.concat([stats_geo, data], ignore_index=True)
            
        os.makedirs(config.pywake_results_settings.dataset_path)
        with open(os.path.join(config.pywake_results_settings.dataset_path, 'wind_park.pkl'), 'wb') as f:
            pickle.dump(temp_wind_park_list, f)
        stats_geo.to_csv(os.path.join(config.pywake_results_settings.dataset_path, 'stats_geo.csv'))
    else:
        with open(os.path.join(config.pywake_results_settings.dataset_path, 'wind_park.pkl'), 'rb') as f:
            temp_wind_park_list = pickle.load(f)
        stats_geo = pd.read_csv(os.path.join(config.pywake_results_settings.dataset_path, 'stats_geo.csv'))

    for i in range(config.pywake_results_settings.num_simulations):
        start_time = time.time()
        sim_res = init_pywake_sim(config, temp_wind_park_list[i], stats_geo.iloc[i], SimpleTurbine())
        flow_map = sim_res.flow_map(HorizontalGrid(x=np.linspace(config.pywake_results_settings.x_range_min, config.pywake_results_settings.x_range_max, config.pywake_results_settings.x_res),
                                                    y=np.linspace(config.pywake_results_settings.y_range_min, config.pywake_results_settings.y_range_max, config.pywake_results_settings.y_res),
                                                    h=1))
        run_time = time.time() - start_time
        
        stats_dict = {
                "Timestamp": datetime.now(),
                "SimulationTime" : run_time,
                "WindFarmModel": flow_map.windFarmModel.__class__.__name__,
                "WakeDeficitModel": flow_map.windFarmModel.wake_deficitModel.__class__.__name__,
                "BlockageDeficitModel": flow_map.windFarmModel.blockage_deficitModel.__class__.__name__,
                "TurbulenceModel": flow_map.windFarmModel.turbulenceModel.__class__.__name__,
                "SuperPositionModel": flow_map.windFarmModel.superpositionModel.__class__.__name__,
                "HubHeight": flow_map.windFarmModel.windTurbines._hub_heights[0],
                "RotorDiameter": flow_map.windFarmModel.windTurbines._diameters[0],
                "WindTurbineModel": flow_map.windFarmModel.windTurbines.__class__.__name__,
                "WindTurbineAmount": len(sim_res.wt),
                "ResX": flow_map.sizes["x"],
                "ResY": flow_map.sizes["y"],
                "WSMin" : float(np.min(flow_map.WS_eff)),
                "WSMax" : float(np.max(flow_map.WS_eff)),
                "WSMean" : float(np.mean(flow_map.WS_eff)),
                "WSStd" : float(np.std(flow_map.WS_eff)),
                "TIMin" : float(np.min(flow_map.TI_eff)),
                "TIMax" : float(np.max(flow_map.TI_eff)),
                "TIMean" : float(np.mean(flow_map.TI_eff)),
                "TIStd" : float(np.min(flow_map.TI_eff)),
                }
        
        stats_sim = pd.concat([stats_sim, pd.DataFrame([stats_dict])], ignore_index=True)

        res_path = os.path.join(config.pywake_results_settings.dataset_path, str(i))
        os.makedirs(res_path)
        sim_res.to_netcdf(os.path.join(res_path, "sim_res.nc"))
        flow_map.to_netcdf(os.path.join(res_path, "flow_map.nc"))
    
    
    stats = pd.merge(stats_geo, stats_sim, left_index=True, right_index=True)
    csv_path = os.path.join(config.pywake_results_settings.dataset_path, "stats.csv")
    # os.makedirs(os.path.dirname(config.pywake_results_settings.dataset_path))
    stats.to_csv(csv_path, index=True) # Save the DataFrame to CSV
    
if __name__ == "__main__":
    config = ConfigLoader.from_yaml('/home/frederikwr/Dropbox/Master_Thesis_public/generate_pywake_sim/config.yaml')
    check_valid_config(config)
    run(config)