#!/usr/bin/env python
# To create relative importance regional bar charts.  >>>  Fig.4d, Fig.5c, SupFig.11c
# By Yusuke Satoh
# On 20200929

import os
import sys
import itertools
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from os.path import join as makepath

hostname = os.uname()[1]
today = datetime.date.today().strftime('%Y%m%d')


# ----------------------------------------------------------------------------------------------------------------------
# find settings below....
setting = sys.argv[1]

#map_type = 'continent'
#map_type = 'ar6_regions'
map_type = sys.argv[2]

#region_type = 'full'; BAR_CHART = False
#region_type = 'selected7'; BAR_CHART = True
region_type = sys.argv[3]
if region_type == 'full':
    BAR_CHART = False
elif region_type == 'selected7':
    BAR_CHART = True
else:
    raise ValueError

anova_dir = 'anova.v3'
#anova_dir = 'anova.final'


# basically you don't need to edit here --------------------------------------------------------------------------------
base = 'base_1861-1960'
version = 'v2'
drought_severity = 'severe'
distribution = 'gamma'

#factors = ['drought_type', 'scenario', 'gcm', 'ghm']
#factors = ['drought_definition', 'scenario', 'gcm', 'ghm']
#factors = ['def', 'scn', 'gcm', 'ghm']
#factors = ['typ', 'scn', 'gcm', 'ghm']

if setting == 'setting1':   # Fig4 main
    MONTHLY = False  # default
    topic_types = ['annual']
    scale_directory = 'scales_full'
    datadate = '20210122'
    factors = ['def', 'scn', 'gcm', 'ghm']
    _regions = ['ENA', 'NEU', 'NEN', 'RAR', 'RFE', 'EAS', 'NWN']  # manually selected top 7 regions
elif setting == 'setting2':  # SupFig9 typ&scl
    MONTHLY = False  # default
    topic_types = ['annual']
    scale_directory = 'scales_full'
    datadate = '20210122'
    factors = ['typ', 'scl', 'scn', 'mdl']
    _regions = ['ENA', 'NEU', 'NEN', 'RAR', 'RFE', 'EAS', 'NWN']  # manually selected top 7 regions
#elif setting == 'setting3':  # Fig5 wheat&maiz
#    MONTHLY = True  # True for crop_calendar
#    monthly_mask_type = 'crop_calendar'
#    topic_types = ['Wheat', 'Maize']
#    scale_directory = 'scale_03'
#    datadate = '20210124'
#    factors = ['def', 'scn', 'gcm', 'ghm']
elif setting == 'setting4':  # Fig5 wheat
    MONTHLY = True  # True for crop_calendar
    monthly_mask_type = 'crop_calendar'
    topic_types = ['Wheat']
    scale_directory = 'scale_03'
    datadate = '20210124'
    factors = ['def', 'scn', 'gcm', 'ghm']
    _regions = ['SAM', 'EAS', 'NEU', 'NAU', 'ECA', 'NES', 'ENA']  # manually selected top 7 regions
elif setting == 'setting5':  # Fig5 Maize
    MONTHLY = True  # True for crop_calendar
    monthly_mask_type = 'crop_calendar'
    topic_types = ['Maize']
    scale_directory = 'scale_03'
    datadate = '20210124'
    factors = ['def', 'scn', 'gcm', 'ghm']
    _regions = ['NEU', 'CNA', 'RFE', 'NES', 'ENA', 'SEAF', 'EAS']  # manually selected top 7 regions
else:
    raise ValueError

climatology = '30yrs'; periods = ['nearfuture', 'farfuture']
#climatology = '50yrs'; periods = ['2nd-half-21C']

factor2combinations = ['{}+{}'.format(*target)    for target in list(itertools.combinations(factors, 2))]
factor3combinations = ['{}+{}+{}'.format(*target) for target in list(itertools.combinations(factors, 3))]
targets_all = factors + factor2combinations + factor3combinations + ['+'.join(factors)]

if 'scs' in hostname:
    data_directory_top = '/data/rg001/sgec0017/data'
else:
    raise ValueError('check hostname... data_cirectory_top cannot be given...')

if MONTHLY and monthly_mask_type == 'crop_calendar': which_directory = 'monthly_crop_calendar'
else:                                                which_directory = 'full'
contributionfrac_directory_main = makepath(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                           anova_dir, which_directory, base, version,
                                           f'climatology_{climatology}',
                                           'anova_level2', drought_severity, distribution, #str(scale),
                                           '_'.join(factors),
                                           scale_directory,
                                           datadate
                                           )
mirca2000_directory = makepath(data_directory_top, 'mapmask', 'MIRCA2000', 'my_edit', 'nc')
gridarea_path = makepath(data_directory_top, 'mapmask', 'grd_ara.hlf')
seamaskpath = makepath(data_directory_top, 'mapmask', 'ISIMIP2b_landseamask_generic.nc4')

outputdirectory_main = makepath(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                'anova-relative_importance', which_directory, base, version,
                                'anova_level2', drought_severity, #distribution, 
                                f'{map_type}_{region_type}',
                                '_'.join(factors),
                                scale_directory,
                                #'3', #str(scale), 
                                today
                                )
if not os.path.isdir(outputdirectory_main): os.makedirs(outputdirectory_main)

dict_topic_types = {
    'Wheat': 1, 'Maize': 2, 'Rice': 3, 'Barley': 4, 'Rye': 5,
    'Millet': 6, 'Sorghum': 7, 'Soybeans': 8, 'Sunflower': 9, 'Potatoes': 10,
    'Cassava': 11, 'SugarCane': 12, 'SugarBeets': 13, 'OilPalm': 14, 'RapeseedCanola': 15,
    'GroundnutsPeanuts': 16, 'Pulses': 17, 'Citrus': 18, 'DatePalm': 19, 'GrapesVine': 20,
    'Cotton': 21, 'Cocoa': 22, 'Coffee': 23, 'OthersPerennial': 24, 'FodderGrasses': 25,
    'OthersAnnual': 26,
    # ---
    'annual': 0,
    }

dict_periods = {
    'nearfuture': '#009e35',  # right green
    'farfuture': '#e6811c',  # orange
    }

ny, nx = 360, 720
suffixes = ['png', 'pdf']

# regional info
if map_type == 'continent':
    regions = ['Global', 'Asia', 'Europe', 'North_America', 'South_America', 'Africa', 'Oceania']
    regionmap_path = makepath(data_directory_top, 'mapmask', 'continent.Kassel.hlf')
    regionmap = np.fromfile(regionmap_path, 'float32').reshape(360, 720)[::-1]
    dict_regions = {
        'Asia': 1,
        'Europe': 2,
        'North_America': 3,
        'South_America': 4,
        'Africa': 5,
        'Oceania': 6,
        }
elif map_type == 'ar6_regions':
    if region_type == 'full':
        regions = [
            'NWN', 'NEN', 'WNA', 'CNA', 'ENA', 
            'NCA', 'SCA', 'CAR', 'NWS', 'NSA', 
            'NES', 'SAM', 'SWS', 'SES', 'SSA', 
            'NEU', 'WCE', 'EEU', 'MED', 'SAH', 
            'WAF', 'CAF', 'NEAF', 'SEAF', 'WSAF',
            'ESAF','MDG', 'RAR', 'WSB', 'ESB',
            'RFE', 'WCA', 'ECA', 'TIB', 'EAS',
            'ARP', 'SAS', 'SEA', 'NAU', 'CAU',
            'EAU', 'SAU', 'NZ',
            ]
    elif region_type == 'selected7':
        regions = ['Global'] + _regions
    else:
        raise ValueError(f'region_type={region_type}')
    regionmap_path = os.path.join(data_directory_top, 'mapmask', 
                                  'IPCC-AR6-WGI-reference-regions-v4_shapefile', 'nc', 
                                  'IPCC-WGI-reference-regions-v4.nc')
    regionmap = Dataset(regionmap_path)['ipcc_regions_ar6'][:]  # includes oceanic regions

    dict_regions = {
        'NWN': 1, 'NEN': 2, 'WNA': 3, 'CNA': 4, 'ENA': 5,
        'NCA': 6, 'SCA': 7, 'CAR': 8, 'NWS': 9, 'NSA': 10,
        'NES': 11, 'SAM': 12, 'SWS': 13, 'SES': 14, 'SSA': 15,
        'NEU': 16, 'WCE': 17, 'EEU': 18, 'MED': 19, 'SAH': 20,
        'WAF': 21, 'CAF': 22, 'NEAF': 23, 'SEAF': 24, 'WSAF': 25,
        'ESAF': 26, 'MDG': 27, 'RAR': 28, 'WSB': 29, 'ESB': 30,
        'RFE': 31, 'WCA': 32, 'ECA': 33, 'TIB': 34, 'EAS': 35,
        'ARP': 36, 'SAS': 37, 'SEA': 38, 'NAU': 39, 'CAU': 40,
        'EAU': 41, 'SAU': 42, 'NZ': 43,
        }

    dict_regionname = {  # manually selected pronounced regions
        'Global': 'Global',
        'CNA': 'C.North-America',
        'EAS': 'E.Asia',
        'ECA': 'E.C.Asia',
        'ENA': 'E.North-America',
        'NAU': 'N.Australia',
        'NEN': 'N.E.North-America',
        'NES': 'N.E.South-America',
        'NEU': 'N.Europe',
        'NWN': 'N.W.North-America',
        'RAR': 'Russian-Arctic',
        'RFE': 'Russian-Far-East',
        'SAM': 'South-American-Monsoon',
        'SEAF': 'S.Eastern-Africa',
        }


# ----------------------------------------------------------------------------------------------------------------------
def load_crop_area_map(crop_type):

    crop_id = dict_topic_types[crop_type]
    filename = f'mirca2000_cropcalendar_{crop_id:02}.{crop_type}_area.nc'
    srcpath = makepath(mirca2000_directory, filename)
    return Dataset(srcpath)['area'][:]


# ----------------------------------------------------------------------------------------------------------------------
def load_input_nc(target, period, topic_type):

    if target == 'unbiasedvariance': target = 'unbiasedvariance_overall'
    elif target == 'frac-std2median': target = 'frac-std2median_overall'
    elif target.count('+') == 0: target = f'frac_0th_main_{target}'
    elif target.count('+') == 1: target = f'frac_1st_interaction_{target}'
    elif target.count('+') == 2: target = f'frac_2nd_interaction_{target}'
    elif target.count('+') == 3: target = 'frac_residual_full'

    if MONTHLY:
        filename = '{}_{}_{:02}.{}_{}_{}.nc'.format(target, period,
                                                    dict_topic_types[topic_type], topic_type,
                                                    distribution, #scale,
                                                    drought_severity)
    else:
        filename = '{}_{}_{}_{}.nc'.format(target, period,
                                           distribution, #scale, 
                                           drought_severity)
    srcpath = makepath(contributionfrac_directory_main, filename)
    print(f'loading... {srcpath}')
    return Dataset(srcpath)[target][0,:]  # (ny, nx)


# ---------------------------------------------------------------------------------------------------------------------
def draw_bar_chart(topic_type, excelpath):

    def gen_xticklabel_item(region):
        if region == 'Global':
            return region
        else:
            return f'{dict_regionname[region]:<}\n({region:<})'

    alpha = 0.9
    bar_width = 0.4
    offset = 0.005
    xticks = np.arange(1, len(regions)+1)
    xticks_f1 = xticks - bar_width/2 - offset
    xticks_f2 = xticks + bar_width/2 + offset
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # --- prepare cmap
    my_colors = np.divide([
        [ 51.,  51., 255.], [191.,   0.,   6.], [169.,  92., 164.], [255., 126.,  51.],
        [142., 186., 217.], [255., 190., 134.], [149., 207., 149.], [234., 146., 147.],  [197., 170., 164.],  [240., 186., 224.],
        [190., 190., 190.], [138., 222., 230.], [189., 227., 255.], [255., 215., 179.],
        [209., 255., 209.],  
        ], 255.)

    if MONTHLY:
        fig = plt.figure(figsize=(8,4.5), dpi=300)
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.11, top=0.95)
        fontsize_large = 12
        fontsize_small = 11
    else:
        fig = plt.figure(figsize=(8.5,4.5), dpi=300)
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.11, top=0.95)
        fontsize_large = 14
        fontsize_small = 11

    ax = fig.add_subplot(111)
    ax.set_title(topic_type, pad=0.02, fontsize=fontsize_large)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if MONTHLY:
        ax.set_xlim([xticks_f1[0]-0.3, xticks_f2[-1]+3.2])
    else:
        ax.set_xlim([xticks_f1[0]-0.3, xticks_f2[-1]+3.10])
    ax.set_xticks(xticks)
    ax.set_xticklabels([region.replace('_', '\n') for region in regions], fontsize=fontsize_large)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize_large)
    ax.grid(axis='y', color='#dbdbdb', linewidth=0.5)
    for iperiod, (_xticks, period) in enumerate(zip([xticks_f1, xticks_f2], periods)):
        df_regional_relativefrac = pd.read_excel(excelpath, index_col=0, sheet_name=f'{topic_type}_{period}')
        for itarget, target in enumerate(targets_all):
            print(itarget, target)
            if iperiod == 0:
                ax.bar(_xticks, df_regional_relativefrac.iloc[itarget, :], bottom=df_regional_relativefrac.iloc[:itarget].sum(),
                       width=bar_width, align='center', color=my_colors[itarget], alpha=alpha, linewidth=1, label=target.replace('+', ',').replace('ghm', 'gwm').replace('typ','ctg'))
            elif iperiod == 1:
                ax.bar(_xticks, df_regional_relativefrac.iloc[itarget, :], bottom=df_regional_relativefrac.iloc[:itarget].sum(),
                       width=bar_width, align='center', color=my_colors[itarget], alpha=alpha, linewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right', frameon=False, fontsize=fontsize_small)
    # save
    figure_name = 'regional_relative_importance_{:02}.{}.'.format(dict_topic_types[topic_type], topic_type)
    figure_path = os.path.join(outputdirectory_main, figure_name)
    for suffix in suffixes:
        plt.savefig(figure_path+suffix, dpi=300)
        print('savefig: {}'.format(figure_path+suffix))
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def main():

    if MONTHLY:
        excelname = 'regional_relativefrac_croparea.xlsx'
    else:
        excelname = 'regional_relativefrac.xlsx'
    excelpath = makepath(outputdirectory_main, excelname)
    writer = pd.ExcelWriter(excelpath)

    for topic_type, period in itertools.product(topic_types, periods):
        df_regional_relativefrac = pd.DataFrame(index=targets_all, columns=regions)

        # --- load input data
        if MONTHLY and monthly_mask_type == 'crop_calendar':
            area = load_crop_area_map(topic_type)  # (ny, nx)  [m2] (note: grids are masked, if there are no crop area there)
        else:
            area = np.fromfile(gridarea_path, 'float32').byteswap().reshape(360, 720)  # [m2] un-masked

        uncertainty_weight = np.abs(load_input_nc('frac-std2median', period, topic_type))  # (ny, nx)  masked

        grid_relativefracs = np.array([load_input_nc(target, period, topic_type) for target in targets_all])  # (11, ny, nx)
        # --- Just for the case...
        if np.any(grid_relativefracs.sum(axis=0)) > 1:
            import warnings
            warnings.warn('\n\nWarning!! total fraction is more than 1. You should check anova.py.\n\n')
            grid_relativefracs = grid_relativefracs * (1/grid_relativefracs.sum(axis=0))
            grid_relativefracs[np.isnan(grid_relativefracs)] = 1e+20
            grid_relativefracs = np.ma.masked_equal(grid_relativefracs, 1e+20)
        # --- preparation
        variance_score = uncertainty_weight * area  # (ny, nx)
        variance_score_frac = grid_relativefracs * variance_score  # (11, ny, nx)

        for region in regions:
            # --- allocate regional mask
            if region == 'Global': regional_mask = Dataset(seamaskpath)['LSM'][:][0].mask
            else:                  regional_mask = np.ma.masked_not_equal(regionmap, dict_regions[region]).mask
            variance_score_regionalmasked = np.ma.masked_array(variance_score, mask=regional_mask)
            variance_score_frac_regionalmasked = np.ma.masked_array(variance_score_frac, mask=np.resize(regional_mask, variance_score_frac.shape))
            # --- write out about each target
            for itarget, target in enumerate(targets_all):
                relativeimportance = variance_score_frac_regionalmasked[itarget].sum() / variance_score_regionalmasked.sum()
                df_regional_relativefrac.loc[target, region] = relativeimportance

        df_regional_relativefrac.to_excel(writer, sheet_name=f'{topic_type}_{period}')

    writer.save()
    writer.close()
    print(f'save: {excelpath}')

    if BAR_CHART:
        for topic_type in topic_types:
            draw_bar_chart(topic_type, excelpath)

    print('This process was successfully DONE!!   d(^o^)b')


if __name__ == '__main__':
    main()
