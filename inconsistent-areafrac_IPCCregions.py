#!/usr/bin/env python
# To estimate area fraction of area with disagreement. This script create bar charts for the top X regions of area fraction with disagreement.  >>> Fig.3e-g, SupFig.8-10
# By Yusuke Satoh
# On 20200929

import os
import sys
import itertools
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from os.path import join as makepath
from netCDF4 import Dataset

hostname = os.uname()[1]
today = datetime.date.today().strftime('%Y%m%d')


# ----------------------------------------------------------------------------------------------------------------------
#scenario = 'rcp26'
#scenario = 'rcp60'
#scenario = 'rcp85'
scenario = sys.argv[1]

MONTHLY = False  # default
#MONTHLY = True # True for crop_calendar

if MONTHLY:
    monthly_mask_type = 'crop_calendar';  topic_types = ['Wheat', 'Maize']; datadate = '20201013'
    #monthly_mask_type = 'crop_calendar'; topic_types = ['Wheat'         ]; datadate = '20201013'
else:
    topic_types = ['annual']; datadate = '20210122'

climatology = '30yrs'; periods = ['nearfuture', 'farfuture']
#climatology = '30yrs'; periods = [ 'farfuture']
#climatology = '50yrs'; periods = ['2nd-half-21C']

Signs = ['-+', '+-', '--', '++']
Robustness = ['SignifAgree', 'Else']
indexes = [f'{signs}{robustness}' for signs in Signs for robustness in Robustness] + ['Disagreement_total', 'Total']

shpfile = gpd.read_file('/data/rg001/sgec0017/data/mapmask/IPCC-AR6-WGI-reference-regions-v4_shapefile/shapefile/IPCC-WGI-reference-regions-v4.shp')
regions = []
dict_regions = {}
for i, (type, name) in enumerate(zip(shpfile['Type'], shpfile['Acronym'])):
    dict_regions[name] = i
    if not type == 'Ocean':
        regions.append(name)
for exeption in ['GIC', 'WAN', 'EAN']:
    regions.remove(exeption)

variables = ['pr', 'qtot', 'soilmoist']
combinations = list(itertools.combinations(variables, 2))

#scales = [1, 3, 6, 12]
#scales = [3]
#scales = [12]
scales = [3, 12]


# fixed parematers --------------------------------------------------------------------------------------------
base = 'base_1861-1960'
version = 'v2'
drought_severity = 'severe'
distribution = 'gamma'

if 'scs' in hostname:
    data_directory_top = '/data/rg001/sgec0017/data'
else:
    raise ValueError('check hostname... data_cirectory_top cannot be given...')

if MONTHLY and monthly_mask_type == 'crop_calendar':
    which_directory = 'monthly_crop_calendar'
else:
    which_directory = 'full'
contributionfrac_directory_main = makepath(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                           'anova', which_directory, base, version,
                                           f'climatology_{climatology}',
                                           'anova_level2', drought_severity, distribution, #str(scale),
                                           datadate
                                               )
mirca2000_directory = makepath(data_directory_top, 'mapmask', 'MIRCA2000', 'my_edit', 'nc')
continent_path = makepath(data_directory_top, 'mapmask', 'continent.Kassel.hlf')
gridarea_path  = makepath(data_directory_top, 'mapmask', 'grd_ara.hlf')
seamaskpath    = makepath(data_directory_top, 'mapmask', 'ISIMIP2b_landseamask_generic.nc4')

outputdirectory_main = makepath(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                'inconsistent-areafrac_IPCCregions', which_directory, base, version,
                                'anova_level2', drought_severity, distribution, #str(scale), 
                                today
                                )
if not os.path.isdir(outputdirectory_main): os.makedirs(outputdirectory_main)

ipcc_ar6_region_path = '/data/rg001/sgec0017/data/mapmask/IPCC-AR6-WGI-reference-regions-v4_shapefile/nc/IPCC-WGI-reference-regions-v4.nc'
ipcc_ar6_region_map = Dataset(ipcc_ar6_region_path)['ipcc_regions_ar6'][:]
seamask = Dataset(seamaskpath)['LSM'][:][0].mask

dict_standerdizedname = {'pr': 'SPI', 'qtot': 'SRI', 'soilmoist': 'SSI'}

dict_sign = {
    '-+': '#efeb48',  # yellow
    '+-': '#2bb8b3',  # bluegreen
    }

ny, nx = 360, 720
suffixes = ['png', 'pdf']


# ----------------------------------------------------------------------------------------------------------------------
def draw_bar_chart(topic_type, excelpath, variable_1, variable_2, period, scale, scenario):

    howmanyregions = 10

    df = pd.read_excel(excelpath, index_col=0)
    # sort by Disagreement_total 
    df = df.T.sort_values(by='Disagreement_total', ascending=False).T
    print(df.T)
    _regions = df.columns[:howmanyregions]

    fontsize_big = 15
    fontsize_small = 13.5
    alpha = 1
    bar_width = 0.4
    edgecolor = '#00000000'
    offset = 0.005
    xticks = np.arange(1, len(_regions)+1)
    xticks_f1 = xticks - bar_width/2 - offset
    xticks_f2 = xticks + bar_width/2 + offset

    fig = plt.figure(figsize=(7.5,3), dpi=300)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.96)
    ax = fig.add_subplot(111)
    ax.text(0.5, 1, f'{}-{}'.format(dict_standerdizedname[variable_1], dict_standerdizedname[variable_2]),
            ha='center', va='top', transform=ax.transAxes, fontsize=fontsize_big)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([xticks_f1[0]-0.3, xticks_f2[-1]+0.3])
    ax.set_xticks(xticks)
    ax.set_xticklabels([region for region in _regions], fontsize=fontsize_big)  #, rotation=90)
    ax.tick_params(axis='y', labelsize=fontsize_small)
    ax.grid(axis='y', color='#dbdbdb', linewidth=0.5)
    for i_sign, (_xticks, sign) in enumerate(zip([xticks_f1, xticks_f2], ['-+', '+-'])):

        ax.bar(_xticks, df.loc[f'{sign}SignifAgree', :_regions[-1]], hatch='///',
               width=bar_width, align='center', color=dict_sign[sign], alpha=alpha, linewidth=0.5, edgecolor=edgecolor)

        ax.bar(_xticks,  df.loc[f'{sign}Else', :_regions[-1]], bottom=df.loc[f'{sign}SignifAgree', :_regions[-1]],
               width=bar_width, align='center', color=dict_sign[sign], alpha=alpha, linewidth=0.5, edgecolor=edgecolor,
               label='{}{}, {}{}'.format(dict_standerdizedname[variable_1],sign[0],dict_standerdizedname[variable_2],sign[1]))
    ax.legend(loc='upper right', frameon=False, fontsize=fontsize_small)
    figure_name = f'inconsistent-areafrac_IPCCregions_{scenario}_{period}_scale{scale:02}_{variable_1}-{variable_2}.'
    figure_path = os.path.join(outputdirectory_main, figure_name)
    for suffix in suffixes:
        plt.savefig(figure_path+suffix, dpi=300)
        print('savefig: {}'.format(figure_path+suffix))
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def load_input_nc(target, period, scale, topic_type):

    if 'inconsisntentSign-variable' in target:
        #src_directory = '/data/rg001/sgec0017/data/figure_box/isimip2b.standardized_drought/basicMaps.separate.find_inconsisnteces_variables/base_1861-1960/v2/climatology_30yrs/20210119/{}/median/{}/{}/{}'.format(scenario, drought_severity, distribution, scale)
        src_directory = '/data/rg001/sgec0017/data/figure_box/isimip2b.standardized_drought/basicMaps.separate.find_inconsisnteces_variables/climatology_30yrs/{}/median/{}/{}/{}'.format(scenario, drought_severity, distribution, scale)
        filename = 'basicMap.ensemblemedian_{}__period_total_drought_months_{}_farfuture__gamma_{}_severe.nc'.format(target, scenario, scale)
        nc_var_name = 'agree-disagree'
    elif 'ksmask' in target:
        variable = target.split('_')[1]
        #src_directory = '/data/rg001/sgec0017/data/figure_box/isimip2b.standardized_drought/basicMaps.separate.output_climetincides/base_1861-1960/v2/climatology_30yrs_KS-ON/20210122/{}/median/{}/{}/{}'.format(scenario, variable, distribution, scale)
        src_directory = '/data/rg001/sgec0017/data/figure_box/isimip2b.standardized_drought/basicMaps.separate.output_climetincides/climatology_30yrs_KS-ON/{}/median/{}/{}/{}'.format(scenario, variable, distribution, scale)
        filename = 'ksmask_{}_{}_median_gamma_{}_severe.nc'.format(variable, period, scale)
        nc_var_name = target
    elif 'agreement' in target:
        variable = target.split('_')[1]
        #src_directory = '/data/rg001/sgec0017/data/figure_box/isimip2b.standardized_drought/basicMaps.separate.output_climetincides/base_1861-1960/v2/climatology_30yrs_KS-ON/20210122/{}/median/{}/{}/{}'.format(scenario, variable, distribution, scale)
        src_directory = '/data/rg001/sgec0017/data/figure_box/isimip2b.standardized_drought/basicMaps.separate.output_climetincides/climatology_30yrs_KS-ON/{}/median/{}/{}/{}'.format(scenario, variable, distribution, scale)
        filename = 'agreement_{}_{}_median_gamma_{}_severe.nc'.format(variable, period, scale)
        nc_var_name = target

    srcpath = os.path.join(src_directory, filename)
    return Dataset(srcpath)[nc_var_name][0]


# ---------------------------------------------------------------------------------------------------------------------
def main():

    for topic_type, period, scale, (variable_1, variable_2) in itertools.product(topic_types, periods, scales, combinations):

        df_regional_relativefrac = pd.DataFrame(index=indexes, columns=regions)
        excelname = f'regional_relativefrac_{scenario}_{period}_scale{scale:02}_{variable_1}-{variable_2}.xlsx'
        excelpath = makepath(outputdirectory_main, excelname)
        writer = pd.ExcelWriter(excelpath)

        # --- load input data
        # grid area
        area = np.fromfile(gridarea_path, 'float32').byteswap().reshape(360, 720)  # [m2] un-masked
        area = np.ma.masked_array(area, mask=seamask)
        # agreemant/disagreement map
        agree_disagree = load_input_nc(f'inconsisntentSign-variable_{variable_1}-{variable_2}', period, scale, topic_type)

        # KS-mask
        ksmask_1 = np.ma.make_mask(load_input_nc(f'ksmask_{variable_1}', period, scale, topic_type) == 1)  # to remove statistically insignificant changes
        ksmask_2 = np.ma.make_mask(load_input_nc(f'ksmask_{variable_2}', period, scale, topic_type) == 1)  # to remove statistically insignificant changes
        ksmask   = np.ma.mask_or(ksmask_1, ksmask_2)
        # ensemble member agreement level
        emalmask_1 = np.ma.masked_less(load_input_nc(f'agreement_{variable_1}', period, scale, topic_type), 0.6).mask  # to remove low agreement changes
        emalmask_2 = np.ma.masked_less(load_input_nc(f'agreement_{variable_2}', period, scale, topic_type), 0.6).mask  # to remove low agreement changes
        emalmask   = np.ma.mask_or(emalmask_1, emalmask_2)
        # ---
        robustchange_mask = np.ma.mask_or(ksmask, emalmask)  # to highlight statistically significant change with high emsemble member agreement
        non_robustchange_mask = ~robustchange_mask  # inverse. to highlight non-robust changes

        for region in regions:
            # --- allocate regional mask
            if region == 'Global':
                regional_mask = Dataset(seamaskpath)['LSM'][:][0].mask
            else:
                regional_mask = np.ma.masked_not_equal(ipcc_ar6_region_map, dict_regions[region]).mask

            area_regionalmasked = np.ma.masked_array(area, mask=regional_mask)
            area_regional_total = area_regionalmasked.sum()

            for signs, robustness in itertools.product(Signs, Robustness):
                index_name = f'{signs}{robustness}'

                # make a mask about a pair of signs of change
                if   signs == '--': signs_mask = np.ma.make_mask(agree_disagree != 0.5)
                elif signs == '-+': signs_mask = np.ma.make_mask(agree_disagree != 1.5)
                elif signs == '++': signs_mask = np.ma.make_mask(agree_disagree != 2.5)
                elif signs == '+-': signs_mask = np.ma.make_mask(agree_disagree != 3.5)
                # make a mask about robustness
                if   robustness == 'SignifAgree': rubstness_mask = robustchange_mask
                elif robustness == 'Else': rubstness_mask = non_robustchange_mask
                # merge maskes
                merged_mask = np.ma.mask_or(signs_mask, rubstness_mask)

                target_area_total = np.ma.masked_array(area_regionalmasked, mask=merged_mask).filled(0).sum()
                area_fraction = target_area_total / area_regional_total

                df_regional_relativefrac.loc[index_name, region] = area_fraction

            df_regional_relativefrac.loc['Disagreement_total', region] = df_regional_relativefrac.loc[indexes[0]:indexes[3], region].fillna(0).sum()
            df_regional_relativefrac.loc['Total', region] = df_regional_relativefrac.loc[indexes[0]:indexes[7], region].fillna(0).sum()

        df_regional_relativefrac.to_excel(writer)
        writer.save()
        writer.close()
        print(f'excel out: {excelpath}')

        draw_bar_chart(topic_type, excelpath, variable_1, variable_2, period, scale, scenario)

    print('This process was successfully DONE!!   d(^o^)b')


if __name__ == '__main__':
    main()
