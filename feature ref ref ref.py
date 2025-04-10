import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt
import scipy.integrate as it
import warnings
import glob
import time
import warnings
import pickle
import sys
import itertools
import ast
import traceback
from sklearn.metrics.pairwise import pairwise_distances
from scipy.integrate import cumulative_trapezoid



def create_dir(file):
    try:
        os.makedirs(file)
    except FileExistsError:
        pass

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)



np.seterr(divide='ignore', invalid='ignore')
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

g = 9.8

thresh = 0.003328*g
fs = 30
thresh_count = 2
dx = 1/fs

fc_hp = 2.5
fc_lp = 0.25


def main():
    print("hi1")
    start_idx, end_idx = linear_movements_detection(1, "Test")
    print(start_idx)
    print(end_idx)

    for split in ['']:
    
        hi = load(start_idx, end_idx, split)
        print(hi)
    
    # args = sys.argv[1]
    
    # id = ast.literal_eval(args.split('_')[0])
    # week = args.split('_')[1]
            
    # linear_movements_detection(id, week)
    
    # for split in ['']:
    
    #     load(id, week, split)
    

def get_zero_crossing(velocity):

    zero_crossing = []

    for i, v in enumerate(velocity[:-1]):
        if (velocity[i+1] >= 0 and velocity[i] <= 0) or (velocity[i+1] <= 0 and velocity[i] >= 0):
            zero_crossing.append(i)
        else:
            continue


    return np.array(zero_crossing)


def get_SM_displacement(vel):
    
    displacement = np.trapz(vel, np.arange(len(vel))/fs)

    return displacement

def normalization(data):
    return (data-np.min(data))/((np.max(data)-np.min(data)))

def get_PC_scores(data):
    
    return np.nanmin(data, axis=0), np.nanmax(data, axis=0), np.nanmean(data, axis=0), np.nanmedian(data, axis=0), np.nanstd(data, axis=0), np.nanpercentile(data, axis=0, q=10), np.nanpercentile(data, axis=0, q=90), iqr(data), kurtosis(np.abs(data), axis=0)


def get_10percentile(data):
    return np.nanpercentile(data, 10)

def get_90percentile(data):
    return np.nanpercentile(data, 90)

def get_kurtosis(data):
    return kurtosis(np.abs(data), axis=0)

def get_features(data):

    return np.nanmin(data), np.nanmax(data), np.nanmean(data), np.nanmedian(data), np.nanstd(data),  get_10percentile(data), get_90percentile(data), iqr(data)

def get_vel_acc_jerk_features(data):
    min_result = np.vectorize(np.nanmin)(data)
    max_result = np.vectorize(np.nanmax)(data)
    mean_result = np.vectorize(np.nanmean)(data)
    median_result = np.vectorize(np.nanmedian)(data)
    std_result = np.vectorize(np.nanstd)(data)
    per10_result = np.vectorize(get_10percentile)(data)
    per90_result = np.vectorize(get_90percentile)(data)
    iqr_result = np.vectorize(iqr)(data)
    kur_result = np.vectorize(get_kurtosis)(data)
    
        
    features = []
    features.extend(get_features(min_result))
    features.extend(get_features(max_result))
    features.extend(get_features(mean_result))
    features.extend(get_features(median_result))
    features.extend(get_features(std_result))
    features.extend(get_features(per10_result))
    features.extend(get_features(per90_result))
    features.extend(get_features(iqr_result))
    features.extend(get_features(kur_result))
    
    return features


def DBSCAN_predict(training_set, testing_set, epsilon, k=5):
    distance_matrix = pairwise_distances(testing_set, training_set)
    sorted_dist_matrix = np.sort(distance_matrix, axis=1)[:, k-2]
    return sorted_dist_matrix <= epsilon


with open('PC1_homo_samples.pkl', 'rb') as PC1_homo_file:
    pc1_homo_samples = pickle.load(PC1_homo_file)

with open('PC2_homo_samples.pkl', 'rb') as PC2_homo_file:
    pc2_homo_samples = pickle.load(PC2_homo_file)

epsilon = 0.375

 
def getBouts(local_acc, start_idx, end_idx):

    activity_bout_acc = []
    activity_bout_jerk = []
    activity_bout_vel = []

    zero_crossing = [[], []]
    # extract the data         

    pca = PCA(n_components=2, random_state=0)

    for i in range(len(start_idx)):

        start = start_idx[i]
        end = end_idx[i]
            
        activity_bout_acc.append(pca.fit_transform(local_acc[start:end]))

        activity_bout_jerk.append(np.gradient(activity_bout_acc[len(activity_bout_acc)-1], dx, axis=0))
        # activity_bout_vel.append(it.cumtrapz(activity_bout_acc[len(activity_bout_acc)-1], dx=dx, axis=0))
        activity_bout_vel.append(it.cumulative_trapezoid(activity_bout_acc[len(activity_bout_acc)-1], dx=dx, axis=0))
        zero_crossing[0].append(get_zero_crossing(activity_bout_vel[len(activity_bout_acc)-1][:, 0]))
        zero_crossing[1].append(get_zero_crossing(activity_bout_vel[len(activity_bout_acc)-1][:, 1]))

    return np.array(activity_bout_acc), np.array(activity_bout_jerk), np.array(activity_bout_vel), np.array(zero_crossing)



def get_SM_features(activity_bout_acc, activity_bout_jerk, activity_bout_vel, zero_crossing):

	PC1_all = {"SM_dis":[], "SM_vel":[], "SM_acc":[], "SM_jerk":[], "SM_duration":[]}
	PC2_all = {"SM_dis":[], "SM_vel":[], "SM_acc":[], "SM_jerk":[], "SM_duration":[]}

	PC1_homo = {"SM_dis":[], "SM_vel":[], "SM_acc":[], "SM_jerk":[], "SM_duration":[]}
	PC2_homo = {"SM_dis":[], "SM_vel":[], "SM_acc":[], "SM_jerk":[], "SM_duration":[]}

	PC1_outlier = {"SM_dis":[], "SM_vel":[], "SM_acc":[], "SM_jerk":[], "SM_duration":[]}
	PC2_outlier = {"SM_dis":[], "SM_vel":[], "SM_acc":[], "SM_jerk":[], "SM_duration":[]}


	PC1_all_resampled_vel = []
	PC2_all_resampled_vel = []

	PC1_homo_resampled_vel = []
	PC2_homo_resampled_vel = []

	PC1_outlier_resampled_vel = []
	PC2_outlier_resampled_vel = []
    
    
	homo_vel = []
	outlier_vel = []
    
    
	pc1_count_all = 0.1
	pc1_count_homo = 0
    
	pc2_count_all = 0.1
	pc2_count_homo = 0

	for bout_count in range(activity_bout_vel.shape[0]):

		direction = 0
		for j in range(1, len(zero_crossing[direction][bout_count])):
			duration = (zero_crossing[direction][bout_count][j]-zero_crossing[direction][bout_count][j-1])/fs

			if (duration > 0.05):
                
				pc1_count_all += 1

				peak_vel = np.max(np.abs((activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction])))

				PC1_all['SM_dis'].append(get_SM_displacement(np.abs(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction])))
				PC1_all['SM_vel'].append(np.abs(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction]))
				PC1_all['SM_acc'].append(activity_bout_acc[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction])
				PC1_all['SM_jerk'].append(activity_bout_jerk[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction])
				PC1_all['SM_duration'].append(duration)

				PC1_all_resampled_vel.append(scipy.signal.resample(normalization(np.abs(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction])), 40))


				vel_resampled = scipy.signal.resample(np.abs(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction]), 15)
				vel_resampled = vel_resampled/np.mean(vel_resampled)

				if (DBSCAN_predict(pc1_homo_samples, vel_resampled.reshape(-1, 15), epsilon, k=5)):
                    
					pc1_count_homo += 1
					PC1_homo['SM_dis'].append(PC1_all['SM_dis'][-1])
					PC1_homo['SM_vel'].append(PC1_all['SM_vel'][-1])
					PC1_homo['SM_acc'].append(PC1_all['SM_acc'][-1])
					PC1_homo['SM_jerk'].append(PC1_all['SM_jerk'][-1])
					PC1_homo['SM_duration'].append(duration)

					PC1_homo_resampled_vel.append(PC1_all_resampled_vel[-1])
				else:
					PC1_outlier['SM_dis'].append(PC1_all['SM_dis'][-1])
					PC1_outlier['SM_vel'].append(PC1_all['SM_vel'][-1])
					PC1_outlier['SM_acc'].append(PC1_all['SM_acc'][-1])
					PC1_outlier['SM_jerk'].append(PC1_all['SM_jerk'][-1])
					PC1_outlier['SM_duration'].append(duration)

					PC1_outlier_resampled_vel.append(PC1_all_resampled_vel[-1])



		direction = 1
		for j in range(1, len(zero_crossing[direction][bout_count])):

			duration = (zero_crossing[direction][bout_count][j]-zero_crossing[direction][bout_count][j-1])/fs

			if (duration > 0.05):
                
				pc2_count_all += 1

				peak_vel = np.max(np.abs(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction]))
				
				PC2_all['SM_dis'].append(get_SM_displacement(np.abs(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction])))

				PC2_all['SM_vel'].append(np.abs(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction]))
				PC2_all['SM_acc'].append((activity_bout_acc[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction]))
				PC2_all['SM_jerk'].append((activity_bout_jerk[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction]))
				PC2_all['SM_duration'].append(duration)

				PC2_all_resampled_vel.append(scipy.signal.resample(normalization(np.abs(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction])), 40))


				vel_resampled = scipy.signal.resample(np.abs(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction]), 10)
				vel_resampled = vel_resampled/np.mean(vel_resampled)


				if (DBSCAN_predict(pc2_homo_samples, vel_resampled.reshape(-1, 10), epsilon, k=5)):
                    
					pc2_count_homo += 1
                        
					# homo_vel.append(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction])
					PC2_homo['SM_dis'].append(PC2_all['SM_dis'][-1])
					PC2_homo['SM_vel'].append(PC2_all['SM_vel'][-1])
					PC2_homo['SM_acc'].append(PC2_all['SM_acc'][-1])
					PC2_homo['SM_jerk'].append(PC2_all['SM_jerk'][-1])
					PC2_homo['SM_duration'].append(duration)

					PC2_homo_resampled_vel.append(PC2_all_resampled_vel[-1])
				else:
					# outlier_vel.append(activity_bout_vel[bout_count][zero_crossing[direction][bout_count][j-1]:zero_crossing[direction][bout_count][j], direction])
					PC2_outlier['SM_dis'].append(PC2_all['SM_dis'][-1])
					PC2_outlier['SM_vel'].append(PC2_all['SM_vel'][-1])
					PC2_outlier['SM_acc'].append(PC2_all['SM_acc'][-1])
					PC2_outlier['SM_jerk'].append(PC2_all['SM_jerk'][-1])
					PC2_outlier['SM_duration'].append(duration)

					PC2_outlier_resampled_vel.append(PC2_all_resampled_vel[-1])


	pca = PCA(n_components=5, random_state=2023)

	try:
		PC1_all_resampled_vel_projected = np.swapaxes(pca.fit_transform(np.array(np.swapaxes(PC1_all_resampled_vel, 0, 1))), 0, 1)
		PC1_homo_resampled_vel_projected = np.swapaxes(pca.fit_transform(np.array(np.swapaxes(PC1_homo_resampled_vel, 0, 1))), 0, 1)
		PC1_outlier_resampled_vel_projected = np.swapaxes(pca.fit_transform(np.array(np.swapaxes(PC1_outlier_resampled_vel, 0, 1))), 0, 1)
	except:
		PC1_all_resampled_vel_projected = np.zeros((1, 5))
		PC1_homo_resampled_vel_projected = np.zeros((1, 5))
		PC1_outlier_resampled_vel_projected = np.zeros((1, 5))

	try:
		PC2_all_resampled_vel_projected = np.swapaxes(pca.fit_transform(np.array(np.swapaxes(PC2_all_resampled_vel, 0, 1))), 0, 1)
		PC2_homo_resampled_vel_projected = np.swapaxes(pca.fit_transform(np.array(np.swapaxes(PC2_homo_resampled_vel, 0, 1))), 0, 1)
		PC2_outlier_resampled_vel_projected = np.swapaxes(pca.fit_transform(np.array(np.swapaxes(PC2_outlier_resampled_vel, 0, 1))), 0, 1)
	except:
		PC2_all_resampled_vel_projected = np.zeros((1, 5))
		PC2_homo_resampled_vel_projected = np.zeros((1, 5))
		PC2_outlier_resampled_vel_projected = np.zeros((1, 5))


	return PC1_all, PC1_homo, PC1_outlier, PC2_all, PC2_homo, PC2_outlier, PC1_all_resampled_vel_projected, PC1_homo_resampled_vel_projected, PC1_outlier_resampled_vel_projected, PC2_all_resampled_vel_projected, PC2_homo_resampled_vel_projected, PC2_outlier_resampled_vel_projected, pc1_count_homo/pc1_count_all, pc2_count_homo/pc2_count_all
    
	# return homo_vel, outlier_vel, PC2_homo


from scipy.stats import linregress, skew, kurtosis
from skimage.measure import regionprops, label
from scipy.spatial import ConvexHull


def iqr(dist):
    return np.nanpercentile(dist, 75) - np.nanpercentile(dist, 25)
    

def compute_ratio(prior, next):
    
    ratio = np.log10(np.absolute(np.array(prior) / np.array(next)))
    ratio = ratio[np.isfinite(ratio)]

    return ratio


def get_vel_acc_jerk_features_temporal(data, acc_or_jerk):
    features = []
    features.extend(get_hisotgram_features(np.vectorize(np.nanmin)(data), acc_or_jerk))
    features.extend(get_hisotgram_features(np.vectorize(np.nanmax)(data), acc_or_jerk))
    features.extend(get_hisotgram_features(np.vectorize(np.nanmean)(data), acc_or_jerk))
    features.extend(get_hisotgram_features(np.vectorize(np.nanmedian)(data), acc_or_jerk))
    features.extend(get_hisotgram_features(np.vectorize(np.nanstd)(data), acc_or_jerk))
    features.extend(get_hisotgram_features(np.vectorize(get_10percentile)(data), acc_or_jerk))
    features.extend(get_hisotgram_features(np.vectorize(get_90percentile)(data), acc_or_jerk))
    features.extend(get_hisotgram_features(np.vectorize(iqr)(data), acc_or_jerk))
    
    return features
    
    
def get_hisotgram_features(data, acc_or_jerk):

	hist, _, _ = np.histogram2d(data[:-1], data[1:], bins=(1000, 1000))
      
	hist = hist / hist.sum()
		
	features = []
		
	features = geometric_features(hist, acc_or_jerk)

	return features


def aggregate_features(quadrant_features, key):
    # Initialize lists to store feature values
    bbox_widths = []
    bbox_heights = []
    aspect_ratios = []
    convex_hull_areas = []
    convex_hull_perimeter = []
    compactnesses = []
    eccentricities = []

    for region_features in quadrant_features:
        if region_features:  # Check if features are available
            for feature_name, feature_value in region_features.items():
                if 'bbox_width' in feature_name:
                    bbox_widths.append(feature_value)
                elif 'bbox_height' in feature_name:
                    bbox_heights.append(feature_value)
                elif 'bbox_aspect_ratio' in feature_name:
                    aspect_ratios.append(feature_value)
                elif 'convex_hull_area' in feature_name:
                    convex_hull_areas.append(feature_value)
                elif 'convex_hull_perimeter' in feature_name:
                    convex_hull_perimeter.append(feature_value)
                elif 'compactness' in feature_name:
                    compactnesses.append(feature_value)
                elif 'eccentricity' in feature_name:
                    eccentricities.append(feature_value)

    # Define a helper function to compute aggregate statistics
    def compute_statistics(values):
        if values:
            values = np.array(values)
            return {
                'min': np.nanmin(values),
                'max': np.nanmax(values),
                'mean': np.nanmean(values),
                'median': np.nanmedian(values),
                'std': np.nanstd(values),
                '10_percentile': np.nanpercentile(values, 10),
                '90_percentile': np.nanpercentile(values, 90),
                'iqr': np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
            }
        else:
            return {
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                '10_percentile': np.nan,
                '90_percentile': np.nan,
                'iqr': np.nan
            }
    
    # Compute aggregated statistics for each feature
    aggregated_features = {
        'bbox_width': compute_statistics(bbox_widths),
        'bbox_height': compute_statistics(bbox_heights),
        'aspect_ratio': compute_statistics(aspect_ratios),
        'convex_hull_area': compute_statistics(convex_hull_areas),
        'convex_hull_perimeter': compute_statistics(convex_hull_perimeter),
        'compactness': compute_statistics(compactnesses),
        'eccentricity': compute_statistics(eccentricities)
    }

    
    return aggregated_features


def geometric_features(histogram_data, acc_or_jerk):
    # Define quadrants
    mid_x, mid_y = histogram_data.shape[0] // 2, histogram_data.shape[1] // 2
    if (acc_or_jerk):
        quadrants = {
            'q1': histogram_data[:mid_x, :mid_y],
            'q2': histogram_data[:mid_x, mid_y:],
            'q3': histogram_data[mid_x:, :mid_y],
            'q4': histogram_data[mid_x:, mid_y:]
        }
    else:
        quadrants = {'q1':histogram_data}

    all_features = {}

    for key, quadrant in quadrants.items():
        labeled_image = label(quadrant > 0)  # Label connected regions
        regions = regionprops(labeled_image)

        quadrant_features = []
        for i, region in enumerate(regions):
        # Extract features for each region
            region_features = {}
            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr
            region_features[f'region_{i}_bbox_width'] = width
            region_features[f'region_{i}_bbox_height'] = height
            region_features[f'region_{i}_bbox_aspect_ratio'] = width / height

            try:
                hull = ConvexHull(region.coords)
                region_features[f'region_{i}_convex_hull_area'] = hull.volume
                region_features[f'region_{i}_convex_hull_perimeter'] = hull.area
                region_features[f'region_{i}_compactness'] = hull.area**2 / hull.volume

            except:
                region_features[f'region_{i}_convex_hull_area'] = 0
                region_features[f'region_{i}_convex_hull_perimeter'] = 0
                region_features[f'region_{i}_compactness'] = 0

            region_features[f'region_{i}_eccentricity'] = region.eccentricity
            quadrant_features.append(region_features)

        all_features[key] = aggregate_features(quadrant_features, key)

    records = []

    for _, stats in all_features.items():
        for _, stat_values in stats.items():
            for _, stat_value in stat_values.items():
                records.append(stat_value)

    return records



columns = []
for i in ['PC1_all', 'PC2_all', 'PC1_homo', 'PC2_homo', 'PC1_outlier', 'PC2_outlier']:
    for j in ['vel', 'acc', 'jerk']:
        for k in ['min', 'max', 'mean', 'median', 'std', '10per', '90per', 'iqr', 'kurtosis']:
            for l in ['min', 'max', 'mean', 'median', 'std', '10per', '90per', 'iqr']:
                columns.append('{}_{}_{}_{}'.format(i, j, k, l))

                
for i in ['PC1_all', 'PC2_all', 'PC1_homo', 'PC2_homo', 'PC1_outlier', 'PC2_outlier']:
    for j in ['distance', 'duration']:
        for k in ['min', 'max', 'mean', 'median', 'std', '10per', '90per', 'iqr']:
            columns.append('{}_{}_{}'.format(i, j, k))
        
                  
for i in ['SM1_all', 'SM2_all', 'SM1_homo', 'SM2_homo', 'SM1_outlier', 'SM2_outlier']:
    for j in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']:
        for k in ['min', 'max', 'mean', 'median', 'std', '10per', '90per', 'iqr', 'kurtosis']:
            columns.append('{}_{}_{}'.format(i, j, k))
    
    

for i in ['PC1_all_2d_hist', 'PC2_all_2d_hist']:
    for j in ['acc', 'jerk']:
        for k in ['min', 'max', 'mean', 'median', 'std', '10per', '90per', 'iqr']:
            for l in ['q1', 'q2', 'q3', 'q4']:
                for m in ['bbox_width', 'bbox_height', 'aspect_ratio', 'convex_hull_area', 'convex_hull_perimeter', 'compactness', 'eccentricity']:
                    for n in ['min', 'max', 'mean', 'median', 'std', '10per', '90per', 'iqr']:
                        columns.append('{}_{}_{}_{}_{}_{}'.format(i, j, k, l, m, n))
                        
                      
                        
for i in ['PC1_all_2d_hist', 'PC2_all_2d_hist']:
    for j in ['vel']:
        for k in ['min', 'max', 'mean', 'median', 'std', '10per', '90per', 'iqr']:
            for l in ['q1']:
                for m in ['bbox_width', 'bbox_height', 'aspect_ratio', 'convex_hull_area', 'convex_hull_perimeter', 'compactness', 'eccentricity']:
                    for n in ['min', 'max', 'mean', 'median', 'std', '10per', '90per', 'iqr']:
                        columns.append('{}_{}_{}_{}_{}_{}'.format(i, j, k, l, m, n))
                        
                        
for i in ['PC1_all_2d_hist', 'PC2_all_2d_hist']:
    for j in ['distance', 'duration']:
            for k in ['q1']:
                for l in ['bbox_width', 'bbox_height', 'aspect_ratio', 'convex_hull_area', 'convex_hull_perimeter', 'compactness', 'eccentricity']:
                    for m in ['min', 'max', 'mean', 'median', 'std', '10per', '90per', 'iqr']:
                        columns.append('{}_{}_{}_{}_{}'.format(i, j, k, l, m))
                        
                                        
                
columns.append('PC1_homo_portion')
columns.append('PC2_homo_portion')

def get_all_features(acc, start_idx, end_idx):


    all_features_list = [] 


    activity_bout_acc, activity_bout_jerk, activity_bout_vel, zero_crossing = getBouts(acc, start_idx, end_idx)

    PC1_all, PC1_homo, PC1_outlier, PC2_all, PC2_homo, PC2_outlier, PC1_all_resampled_vel_projected, PC1_homo_resampled_vel_projected, PC1_outlier_resampled_vel_projected, PC2_all_resampled_vel_projected, PC2_homo_resampled_vel_projected, PC2_outlier_resampled_vel_projected, PC1_homo_portion, PC2_homo_portion = get_SM_features(activity_bout_acc, activity_bout_jerk, activity_bout_vel, zero_crossing)

    try:
        all_features_list.extend(get_vel_acc_jerk_features(PC1_all['SM_vel']))
        all_features_list.extend(get_vel_acc_jerk_features(PC1_all['SM_acc']))
        all_features_list.extend(get_vel_acc_jerk_features(PC1_all['SM_jerk']))
    except:
        all_features_list.extend(np.zeros(216))


    try:
        all_features_list.extend(get_vel_acc_jerk_features(PC2_all['SM_vel']))
        all_features_list.extend(get_vel_acc_jerk_features(PC2_all['SM_acc']))
        all_features_list.extend(get_vel_acc_jerk_features(PC2_all['SM_jerk']))
    except:
        all_features_list.extend(np.zeros(216)*np.nan)

    
    try:
        all_features_list.extend(get_vel_acc_jerk_features(PC1_homo['SM_vel']))
        all_features_list.extend(get_vel_acc_jerk_features(PC1_homo['SM_acc']))
        all_features_list.extend(get_vel_acc_jerk_features(PC1_homo['SM_jerk']))
    except:
        all_features_list.extend(np.zeros(216)*np.nan)


    try:
        all_features_list.extend(get_vel_acc_jerk_features(PC2_homo['SM_vel']))
        all_features_list.extend(get_vel_acc_jerk_features(PC2_homo['SM_acc']))
        all_features_list.extend(get_vel_acc_jerk_features(PC2_homo['SM_jerk']))
    except:
        all_features_list.extend(np.zeros(216)*np.nan)
    

    try:
        all_features_list.extend(get_vel_acc_jerk_features(PC1_outlier['SM_vel']))
        all_features_list.extend(get_vel_acc_jerk_features(PC1_outlier['SM_acc']))
        all_features_list.extend(get_vel_acc_jerk_features(PC1_outlier['SM_jerk']))
    except:
        all_features_list.extend(np.zeros(216)*np.nan)
        

    try:
        all_features_list.extend(get_vel_acc_jerk_features(PC2_outlier['SM_vel']))
        all_features_list.extend(get_vel_acc_jerk_features(PC2_outlier['SM_acc']))
        all_features_list.extend(get_vel_acc_jerk_features(PC2_outlier['SM_jerk']))
    except:
        all_features_list.extend(np.zeros(216)*np.nan)

    
    try:
        all_features_list.extend(get_features(PC1_all['SM_dis']))
        all_features_list.extend(get_features(PC1_all['SM_duration']))
    except:
        all_features_list.extend(np.zeros(16)*np.nan)


    try:
        all_features_list.extend(get_features(PC2_all['SM_dis']))
        all_features_list.extend(get_features(PC2_all['SM_duration']))
    except:
        all_features_list.extend(np.zeros(16)*np.nan)


    try:
        all_features_list.extend(get_features(PC1_homo['SM_dis']))
        all_features_list.extend(get_features(PC1_homo['SM_duration']))
    except:
        all_features_list.extend(np.zeros(16)*np.nan)


    try:
        all_features_list.extend(get_features(PC2_homo['SM_dis']))
        all_features_list.extend(get_features(PC2_homo['SM_duration']))
    except:
        all_features_list.extend(np.zeros(16)*np.nan)


    try:
        all_features_list.extend(get_features(PC1_outlier['SM_dis']))
        all_features_list.extend(get_features(PC1_outlier['SM_duration']))
    except:
        all_features_list.extend(np.zeros(16)*np.nan)


    try:
        all_features_list.extend(get_features(PC2_outlier['SM_dis']))
        all_features_list.extend(get_features(PC2_outlier['SM_duration']))
    except:
        all_features_list.extend(np.zeros(16)*np.nan)
        

    try:
        all_features_list.extend(get_PC_scores(PC1_all_resampled_vel_projected[:, 0]))
        all_features_list.extend(get_PC_scores(PC1_all_resampled_vel_projected[:, 1]))
        all_features_list.extend(get_PC_scores(PC1_all_resampled_vel_projected[:, 2]))
        all_features_list.extend(get_PC_scores(PC1_all_resampled_vel_projected[:, 3]))
        all_features_list.extend(get_PC_scores(PC1_all_resampled_vel_projected[:, 4]))
    except:
        all_features_list.extend(np.zeros(45)*np.nan)
        
        
    try:
        all_features_list.extend(get_PC_scores(PC2_all_resampled_vel_projected[:, 0]))
        all_features_list.extend(get_PC_scores(PC2_all_resampled_vel_projected[:, 1]))
        all_features_list.extend(get_PC_scores(PC2_all_resampled_vel_projected[:, 2]))
        all_features_list.extend(get_PC_scores(PC2_all_resampled_vel_projected[:, 3]))
        all_features_list.extend(get_PC_scores(PC2_all_resampled_vel_projected[:, 4]))
    except:
        all_features_list.extend(np.zeros(45)*np.nan)
        
        
    try:
        all_features_list.extend(get_PC_scores(PC1_homo_resampled_vel_projected[:, 0]))
        all_features_list.extend(get_PC_scores(PC1_homo_resampled_vel_projected[:, 1]))
        all_features_list.extend(get_PC_scores(PC1_homo_resampled_vel_projected[:, 2]))
        all_features_list.extend(get_PC_scores(PC1_homo_resampled_vel_projected[:, 3]))
        all_features_list.extend(get_PC_scores(PC1_homo_resampled_vel_projected[:, 4]))
    except:
        all_features_list.extend(np.zeros(45)*np.nan)
        
        
    try:
        all_features_list.extend(get_PC_scores(PC2_homo_resampled_vel_projected[:, 0]))
        all_features_list.extend(get_PC_scores(PC2_homo_resampled_vel_projected[:, 1]))
        all_features_list.extend(get_PC_scores(PC2_homo_resampled_vel_projected[:, 2]))
        all_features_list.extend(get_PC_scores(PC2_homo_resampled_vel_projected[:, 3]))
        all_features_list.extend(get_PC_scores(PC2_homo_resampled_vel_projected[:, 4]))
    except:
        all_features_list.extend(np.zeros(45)*np.nan)
        

    try:
        all_features_list.extend(get_PC_scores(PC1_outlier_resampled_vel_projected[:, 0]))
        all_features_list.extend(get_PC_scores(PC1_outlier_resampled_vel_projected[:, 1]))
        all_features_list.extend(get_PC_scores(PC1_outlier_resampled_vel_projected[:, 2]))
        all_features_list.extend(get_PC_scores(PC1_outlier_resampled_vel_projected[:, 3]))
        all_features_list.extend(get_PC_scores(PC1_outlier_resampled_vel_projected[:, 4]))
    except:
        all_features_list.extend(np.zeros(45)*np.nan)
        
    
    try:
        all_features_list.extend(get_PC_scores(PC2_outlier_resampled_vel_projected[:, 0]))
        all_features_list.extend(get_PC_scores(PC2_outlier_resampled_vel_projected[:, 1]))
        all_features_list.extend(get_PC_scores(PC2_outlier_resampled_vel_projected[:, 2]))
        all_features_list.extend(get_PC_scores(PC2_outlier_resampled_vel_projected[:, 3]))
        all_features_list.extend(get_PC_scores(PC2_outlier_resampled_vel_projected[:, 4]))
    except:
        all_features_list.extend(np.zeros(45)*np.nan)


    try:
        all_features_list.extend(get_vel_acc_jerk_features_temporal(PC1_all['SM_acc'], True))
        all_features_list.extend(get_vel_acc_jerk_features_temporal(PC1_all['SM_jerk'], True))
    except:
        all_features_list.extend(np.zeros(3584)*np.nan)
        

    try:
        all_features_list.extend(get_vel_acc_jerk_features_temporal(PC2_all['SM_acc'], True))
        all_features_list.extend(get_vel_acc_jerk_features_temporal(PC2_all['SM_jerk'], True))
    except:
        all_features_list.extend(np.zeros(3584)*np.nan)
        
    
    
    try:
        all_features_list.extend(get_vel_acc_jerk_features_temporal(PC1_all['SM_vel'], False))
    except:
        all_features_list.extend(np.zeros(448)*np.nan)
        
        
    try:
        all_features_list.extend(get_vel_acc_jerk_features_temporal(PC2_all['SM_vel'], False))
    except:
        all_features_list.extend(np.zeros(448)*np.nan)
    
    
    
    try:
        all_features_list.extend(get_hisotgram_features(PC1_all['SM_dis'], False))
        all_features_list.extend(get_hisotgram_features(PC1_all['SM_duration'], False))
    except:
        all_features_list.extend(np.zeros(112)*np.nan)
        
        
    try:
        all_features_list.extend(get_hisotgram_features(PC2_all['SM_dis'], False))
        all_features_list.extend(get_hisotgram_features(PC2_all['SM_duration'], False))
    except:
        all_features_list.extend(np.zeros(112)*np.nan)


    try:
        all_features_list.append(PC1_homo_portion)
        all_features_list.append(PC2_homo_portion)
    except:
        all_features_list.extend(np.zeros(2)*np.nan)

    df_single_subject = pd.DataFrame(columns=columns, index=['0'])
    df_single_subject[columns] = all_features_list
                               

    return df_single_subject


var_sys = np.nanmean(np.load("system_variance.npy"))

def calculate_rel_AI(acc, var_sys):

    rel_AI = np.zeros(shape=(int(acc.shape[0]/fs)))

    for i in range(int(acc.shape[0]/fs)):

        val = 0
        for j in range(3):
            val += ((np.var(acc[i*fs:(i+1)*fs, j]) - var_sys)/var_sys)

        val /= 3

        rel_AI[i] = np.sqrt(val) if val > 0 else 0

    return rel_AI


def get_linear_movements(local_acc):
    # extract the data
    length = len(local_acc)

    sample_rate = fs

    start = 0
    lower = 1
    upper = 5
    lower_bound = int(sample_rate * lower)
    upper_bound = int(sample_rate * upper)

    stride = 1

    pca_thresh = 0.82
    inactivity_thresh = 0.0045

    starts = []
    ends = []

    while start < length - lower_bound:
        end = start + lower_bound

        # calculate activity index
        ai = calculate_rel_AI(local_acc[start:end], var_sys)

        # if the new window is not active
        if np.nanmean(ai) < inactivity_thresh:
            start += stride
            continue

        # calculate pca
        pca = PCA(n_components=1, random_state=0).fit(local_acc[start:end])
        ratio = pca.explained_variance_ratio_[0]
        # if the pca ratio is smaller than the threshold
        if ratio < pca_thresh:
            start += stride
            continue

        for end in range(start+lower_bound*2, start+upper_bound+1, lower_bound):
            # calculate activity_index
            ai = calculate_rel_AI(local_acc[start:end], var_sys)
            # if the new window is not active
            if np.nanmean(ai[-1]) < inactivity_thresh:
                end -= lower_bound
                break

            pca = PCA(n_components=1, random_state=0).fit(local_acc[start:end])
            ratio = pca.explained_variance_ratio_[0]
            # if the pca ratio is smaller than the threshold
            if ratio < pca_thresh:
                end -= lower_bound
                break

        # update start
        starts.append(start)
        ends.append(end)

        start = end


    return np.array(starts), np.array(ends)


def mask(v):
    return int(int(v/54000)%2)
    


def load(start_idx, end_idx, split):
    
    
    folder = 'all' if (split == '') else split.replace('_', '')
    
    # if os.path.exists('features/{}/{}_{}{}.csv'.format(folder, id, week, split)):
    #     return
    
    try:
        
        # df_all = pd.DataFrame(columns=columns)
        # df_meta = pd.read_csv('Data/All patients meta.csv')


        # if (id < 100):
        #     raw_file = glob.glob('../../work/pi_sunghoonlee_umass_edu/Ryan/Referent/Case{}/*{}*.csv'.format(id, week))[0]
        # elif (id < 1000):
        #     affected_limb = 'RUE' if df_meta.loc[df_meta['ID'] == id, 'AffectedSide'].values[0] == 'Right' else 'LUE'
        #     raw_file = glob.glob('../../work/pi_sunghoonlee_umass_edu/Ryan/Acute patients/{}/Week{}/{}/*.csv'.format(id, week, affected_limb))[0]

        #     if not (affected_limb == raw_file.split('/')[-2]):
        #         return
        # else:
        #     affected_limb = 'RUE' if df_meta.loc[df_meta['ID'] == id, 'AffectedSide'].values[0] == 'Right' else 'LUE'
        #     raw_file = glob.glob('../../work/pi_sunghoonlee_umass_edu/Ryan/Chronic patients/{}/{}*/{}/*RAW.csv'.format(id, week, affected_limb))[0]

        #     if not (affected_limb == raw_file.split('/')[-2]):
        #         return
        SCRIPT_DIR = r"C:\AHHA Lab\AHHA Pipeline"
        raw_file = os.path.join(SCRIPT_DIR, "real_data.csv")
        acc = np.array(pd.read_csv(raw_file, skiprows=10, header=None, low_memory=False))


        if (acc[0][0] == 'Accelerometer X'):
            acc = acc[1:].astype(float)

        b, a = butter(6, [fc_lp/(fs/2), fc_hp/(fs/2)], btype='bandpass')
        
        acc = filtfilt(b, a, acc, axis=0)
        
        if ("TW" in raw_file or "Baseline" in raw_file):
            acc = acc[90*60*fs:]
            
        if ("Case" in raw_file):
            acc = acc[60*60*fs:]
        
        # if (id < 100):
        #     start_idx = np.load('../../work/pi_sunghoonlee_umass_edu/Ryan/referent_p2p_idx/{}_{}_start.npy'.format(id, week))
        #     end_idx = np.load('../../work/pi_sunghoonlee_umass_edu/Ryan/referent_p2p_idx/{}_{}_end.npy'.format(id, week))

        # elif (id < 1000):
        #     start_idx = np.load('../../work/pi_sunghoonlee_umass_edu/Ryan/acute_p2p_idx/{}_{}_start.npy'.format(id, week))
        #     end_idx = np.load('../../work/pi_sunghoonlee_umass_edu/Ryan/acute_p2p_idx/{}_{}_end.npy'.format(id, week))

        # else:
        #     start_idx = np.load(glob.glob('../../work/pi_sunghoonlee_umass_edu/Ryan/chronic_p2p_idx/{}_{}_start.npy'.format(id, week))[0])
        #     end_idx = np.load(glob.glob('../../work/pi_sunghoonlee_umass_edu/Ryan/chronic_p2p_idx/{}_{}_end.npy'.format(id, week))[0])
            

        if (split == '_even'):
            mask_array = np.array(list(map(mask, start_idx)))

            start_idx = start_idx[np.where(mask_array == 0)[0]]
            end_idx = end_idx[np.where(mask_array == 0)[0]]

        elif (split == '_odd'):
            mask_array = np.array(list(map(mask, start_idx)))

            start_idx = start_idx[np.where(mask_array == 1)[0]]
            end_idx = end_idx[np.where(mask_array == 1)[0]]

        else:
            pass

            
        df_single_subject = get_all_features(acc, start_idx, end_idx)

        np.set_printoptions(threshold=np.inf)
        features_array = df_single_subject.to_numpy()
        np.set_printoptions(suppress=False, precision=8, linewidth=200)

        # y_pred = model.predict(features_array)
        return features_array
        return df_single_subject.to_numpy()    
    
    except Exception as e:
        with open('{}_{}.log'.format(id, week), 'w') as file_open:

            file_open.write(f"An error occurred: {str(traceback.format_exc())}\n")
            
            
            
def linear_movements_detection(id, week):
      
    try:
        
        # df_all = pd.DataFrame(columns=columns)
        # df_meta = pd.read_csv('Data/All patients meta.csv')


        # if (id < 100):
        #     raw_file = glob.glob('../../work/pi_sunghoonlee_umass_edu/Ryan/Referent/Case{}/*{}*.csv'.format(id, week))[0]
        # elif (id < 1000):
        #     affected_limb = 'RUE' if df_meta.loc[df_meta['ID'] == id, 'AffectedSide'].values[0] == 'Right' else 'LUE'
        #     raw_file = glob.glob('../../work/pi_sunghoonlee_umass_edu/Ryan/Acute patients/{}/Week{}/{}/*.csv'.format(id, week, affected_limb))[0]

        #     if not (affected_limb == raw_file.split('/')[-2]):
        #         return
        # else:
        #     affected_limb = 'RUE' if df_meta.loc[df_meta['ID'] == id, 'AffectedSide'].values[0] == 'Right' else 'LUE'
        #     raw_file = glob.glob('../../work/pi_sunghoonlee_umass_edu/Ryan/Chronic patients/{}/{}*/{}/*RAW.csv'.format(id, week, affected_limb))[0]

        #     if not (affected_limb == raw_file.split('/')[-2]):
        #         return
        print("hi1")
        SCRIPT_DIR = r"C:\AHHA Lab\AHHA Pipeline"
        raw_file = os.path.join(SCRIPT_DIR, "real_data.csv")
        acc = np.array(pd.read_csv(raw_file, skiprows=10, header=None, low_memory=False))

        print("hi1")
        if (acc[0][0] == 'Accelerometer X'):
            acc = acc[1:].astype(float)

        b, a = butter(6, [fc_lp/(fs/2), fc_hp/(fs/2)], btype='bandpass')
        
        acc = filtfilt(b, a, acc, axis=0)
        print("hi2")
        if ("TW" in raw_file or "Baseline" in raw_file):
            acc = acc[90*60*fs:]
            
        if ("Case" in raw_file):
            acc = acc[60*60*fs:]
        
        start_idx, end_idx = get_linear_movements(acc)
        # if (id < 100):
            
        #     start_idx, end_idx = get_linear_movements(acc)

        #     np.save('../../work/pi_sunghoonlee_umass_edu/Ryan/referent_p2p_idx/{}_{}_start.npy'.format(id, week), start_idx)
        #     np.save('../../work/pi_sunghoonlee_umass_edu/Ryan/referent_p2p_idx/{}_{}_end.npy'.format(id, week), end_idx)

        
        # elif (id < 1000):
        #     start_idx, end_idx = get_linear_movements(acc)

        #     np.save('../../work/pi_sunghoonlee_umass_edu/Ryan/acute_p2p_idx/{}_{}_start.npy'.format(id, week), start_idx)
        #     np.save('../../work/pi_sunghoonlee_umass_edu/Ryan/acute_p2p_idx/{}_{}_end.npy'.format(id, week), end_idx)
        
        
        # else:
        
        #     start_idx, end_idx = get_linear_movements(acc)

        #     np.save('../../work/pi_sunghoonlee_umass_edu/Ryan/chronic_p2p_idx/{}_{}_start.npy'.format(id, week), start_idx)
        #     np.save('../../work/pi_sunghoonlee_umass_edu/Ryan/chronic_p2p_idx/{}_{}_end.npy'.format(id, week), end_idx)


        return start_idx, end_idx
    
    except Exception as e:
        with open('{}_{}.log'.format(id, week), 'w') as file_open:

            file_open.write(f"An error occurred: {str(traceback.format_exc())}\n")
    

if __name__ == "__main__":
    main()
        