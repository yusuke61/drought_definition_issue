#!/usr/bin/env python
# For ANOVA analysis  >>> Fig.4a-c, Fig5b, SupFig.11-13
# By Yusuke Satoh (NIES)

import sys
import os
import itertools
import datetime
import numpy as np
import mpl_toolkits.axes_grid1
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join as genpath
from datetime import date
from matplotlib import cm, colors, gridspec
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from utiltools import flipud

hostname = os.uname()[1]
today = datetime.date.today().strftime('%Y%m%d')


# ----- Setting ------------------------------------------------------------------------------------------------------

TEST = False  # default
#TEST = True

DUMMY = False

#REUSE = False
#REUSE = True; date_for_reuse = '20210116'
#REUSE = True; date_for_reuse = '20201025'  # Wheat
#REUSE = True; date_for_reuse = '20201023'  # Maise

#MONTHLY = False  # default
#MONTHLY = True # True for crop_calendar

#factors = ['def', 'scn', 'gcm', 'ghm']  # def = typ x scale
#factors = ['typ', 'scl', 'scn', 'mdl']  # mdl = gcm x ghm
#factors = ['typ', 'scn', 'gcm', 'ghm']  # skip scale  (no longer used)

#periods = ['nearfuture', 'farfuture']
#periods = ['farfuture', 'nearfuture']
#periods = ['nearfuture']
#periods = ['farfuture']

#scales = [1, 3, 6, 12]
#scales = [3]


#anova_version = 'anova.v1'
#anova_version = 'anova.v2
anova_version = 'anova.v3'
#anova_version = 'anova.final'

setting = sys.argv[1]
#setting = 'setting1'  # Fig4
#setting = 'setting2'  # SupFig11
#setting = 'setting3'
#setting = 'setting4'  # SupFig9
#setting = 'setting5'
#setting = 'setting6'  # Wheat  Fig5
#setting = 'setting7'  # Maize  Fig5


if setting == 'setting1':
    #REUSE = False
    REUSE = True; date_for_reuse = '20210122'
    MONTHLY = False
    factors = ['def', 'scn', 'gcm', 'ghm']
    scales = [1, 3, 6, 12]
    periods = ['farfuture'] 
elif setting == 'setting2':
    #REUSE = False
    REUSE = True; date_for_reuse = '20210122'
    MONTHLY = False
    factors = ['def', 'scn', 'gcm', 'ghm']
    scales = [1, 3, 6, 12]
    periods = ['nearfuture']
elif setting == 'setting3':  # with only scale3 nearfuture
    #REUSE = False
    REUSE = True; date_for_reuse = '20210122'
    MONTHLY = False
    factors = ['def', 'scn', 'gcm', 'ghm']
    scales = [3]  # only this one...
    periods = ['nearfuture']
elif setting == 'setting4':  # new factors, farfuture
    #REUSE = False
    REUSE = True; date_for_reuse = '20210122'
    MONTHLY = False
    factors = ['typ', 'scl', 'scn', 'mdl']
    scales = [1, 3, 6, 12]
    periods = ['farfuture'] 
elif setting == 'setting5':  # new factors, nearfuture
    #REUSE = False
    REUSE = True; date_for_reuse = '20210122'
    MONTHLY = False
    factors = ['typ', 'scl', 'scn', 'mdl']
    scales = [1, 3, 6, 12]
    periods = ['nearfuture']
elif setting == 'setting6':  # crop for new factors, nearfuture
    #REUSE = False
    REUSE = True; date_for_reuse = '20210124'
    MONTHLY = True; crop_types = 'Wheat'
    factors = ['def', 'scn', 'gcm', 'ghm']
    scales = [3]
    periods = ['farfuture']
    #periods = ['nearfuture']
elif setting == 'setting7':  # crop for new factors, nearfuture
    #REUSE = False
    REUSE = True; date_for_reuse = '20210124'
    MONTHLY = True; crop_types = 'Maize'
    factors = ['def', 'scn', 'gcm', 'ghm']
    scales = [3]
    periods = ['farfuture']
    #periods = ['nearfuture']

variables = ['pr', 'qtot', 'soilmoist']
scenarios = ['rcp26','rcp60','rcp85']
gcms = ['hadgem2-es','ipsl-cm5a-lr','gfdl-esm2m','miroc5']
ghms = ['matsiro', 'cwatm', 'clm45', 'h08', 'jules-w1', 'lpjml', 'watergap2']
climatology = '30yrs'

if TEST == True:
    #test_name = ''
    variables = ['pr', 'qtot', 'soilmoist']
    scales = [1, 6, 12]; test_name = 'without_scale3'
    scenarios  = ['rcp26','rcp60','rcp85']
    gcms  = ['hadgem2-es','ipsl-cm5a-lr','gfdl-esm2m','miroc5']
    ghms = ['matsiro', 'cwatm', 'clm45', 'h08', 'jules-w1', 'lpjml', 'watergap2']
elif TEST == False:
    pass
else:
    raise ValueError('check argv[1]...')


# ------- Basically, fixed parameters... -------------------------------------------------------------------------------
#anova_level = 1  # like Lehner et al. 2020, Hawkins&Sutton 2009
anova_level = 2  # Hatterman 2018  <--- We will use this!!
#anova_level = 3  # Fractional ANOVA (F-value)

if MONTHLY:
    monthly_mask_type = 'crop_calendar'

# fixed standerdized drought index parameters  (input)
base = 'base_1861-1960'
version = 'v2'
distribution = 'gamma'
drought_severity = 'severe'

suffixes = ['.png', '.pdf']


# ----------------------------------------------------------------------------------------------------------------------
if len(scales) == 1:
   scale_directory = 'scale_{:02}'.format(scales[0])
else:
   scale_directory = 'scales_full'   
factors_directory = '_'.join(factors)

if 'scs' in hostname or 'scfrs' in hostname:
    data_directory_top = '/data/rg001/sgec0017/data'
else:
    raise ValueError('check hostname... data_cirectory_top cannot be given...')
if MONTHLY:
    data_directory_main = os.path.join(data_directory_top, 'isimip2b.standardized_drought',
                                       f'climate_indices_postprocessed_monthly_{monthly_mask_type}',
                                       base, version, 'climatology_{}'.format(climatology))
    if TEST:
        figure_directory_main = os.path.join(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                             anova_version, f'monthly_{monthly_mask_type}_test_{test_name}',
                                             base, version, f'anova_level{anova_level}')
    else:
        figure_directory_main = os.path.join(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                             anova_version, f'monthly_{monthly_mask_type}', base, version,
                                             f'climatology_{climatology}',
                                             f'anova_level{anova_level}')
else:
    data_directory_main = os.path.join(data_directory_top, 'isimip2b.standardized_drought',
                                       'climate_indices_postprocessed', 
                                       base, version, f'climatology_{climatology}')
    if TEST:
        figure_directory_main = os.path.join(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                             anova_version, f'full_test_{test_name}',
                                             base, version, f'climatology_{climatology}'.format(),
                                             f'anova_level{anova_level}')
    else:
        figure_directory_main = os.path.join(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                             anova_version, 'full', base, version, f'climatology_{climatology}',
                                             f'anova_level{anova_level}')
    print(f'figure_directory_main = {figure_directory_main}')
if not os.path.isdir(figure_directory_main): os.makedirs(figure_directory_main)
output_directory = genpath(figure_directory_main, drought_severity, distribution, factors_directory, scale_directory, today)
if not os.path.isdir(output_directory): os.makedirs(output_directory)
figure_directory = output_directory
#figure_directory = genpath(figure_directory_main, drought_severity, distribution, factors_directory, scale_directory, today)
#if not os.path.isdir(figure_directory): os.makedirs(figure_directory)

seamaskpath = genpath(data_directory_top, 'mapmask', 'ISIMIP2b_landseamask_generic.nc4')
grlmskpath = os.path.join(data_directory_top, 'mapmask', 'GAUL/flt/gaul2014_05deg.flt')      # GreenLand is 98

seamask = Dataset(seamaskpath)['LSM'][:][0].mask
grl_mask = np.ma.masked_equal(np.fromfile(grlmskpath, 'float32').reshape(360,720),98).mask
seamask = np.ma.mask_or(seamask, grl_mask)

factor2combinations = list(itertools.combinations(factors, 2))
factor3combinations = list(itertools.combinations(factors, 3))

ndefinition = len(variables)*len(scales)
ntype = len(variables)
nscale = len(scales)
nscenario = len(scenarios)
ngcm = len(gcms)
nghm = len(ghms)
nmdl = ngcm*nghm
ny, nx = 360, 720

if factors == ['def', 'scn', 'gcm', 'ghm']:
    dict_n_factormembers = {'def': ndefinition, 'scn': nscenario, 'gcm': ngcm, 'ghm': nghm}
elif factors == ['typ', 'scl', 'scn', 'mdl']:
    dict_n_factormembers = {'typ': ntype, 'scl': nscale, 'scn': nscenario, 'mdl': nmdl}
else:
    raise ValueError(f'Oh... factors={factors}')

syear, eyear = 1861, 2099
years = range(syear, eyear+1)
nyear = len(years)

dict_soc = {      # hist      rcp60
    'matsiro':    ('histsoc', '2005soc'),
    'clm45':      ('2005soc', '2005soc'),
    'cwatm':      ('histsoc', '2005soc'),
    'h08':        ('histsoc', '2005soc'),
    'jules-w1':   ('nosoc'  , 'nosoc'  ),
    'lpjml':      ('histsoc', '2005soc'),
    'orchidee':   ('nosoc'  , 'nosoc'  ),
    'pcr-globwb': ('histsoc', '2005soc'),
    'watergap2':  ('histsoc', '2005soc'),
    }

if climatology == '30yrs':
    _periods = ['base_period', 'recent30yrs', 'nearfuture', 'farfuture']
    dict_period = {
        'base_period': (1861, 1890),  # 30yrs
        'recent30yrs': (1990, 2019),  # 30yrs
        'nearfuture':  (2035, 2064),  # 30yrs
        'farfuture':   (2070, 2099),  # 30yrs
        }
elif climatology == '50yrs':
    _periods = ['base_period', '1st-half-21C', '2nd-half-21C']
    dict_period = {
        'base_period':  (1861, 1890),  # 50yrs
        '1st-half-21C': (2000, 2049),  # 50yrs
        '2nd-half-21C': (2050, 2099),  # 50yrs
        }
years_full_period = range(1861, 2099+1)

if MONTHLY:
    if monthly_mask_type == 'crop_calendar':
        if crop_types == 'full':
            dict_crop_types = {
                 1: 'Wheat',
                 2: 'Maize',
                 3: 'Rice', 4: 'Barley', 5: 'Rye',
                 6: 'Millet', 7: 'Sorghum', 8: 'Soybeans', 9: 'Sunflower', 10: 'Potatoes',
                 11: 'Cassava', 12: 'SugarCane', 13: 'SugarBeets', 14: 'OilPalm', 15: 'RapeseedCanola',
                 16: 'GroundnutsPeanuts', 17: 'Pulses', 18: 'Citrus', 19: 'DatePalm', 20: 'GrapesVine',
                 21: 'Cotton', 22: 'Cocoa', 23: 'Coffee', 24: 'OthersPerennial', 25: 'FodderGrasses', 26: 'OthersAnnual'
                 }
        elif crop_types == 'Wheat': dict_crop_types = {1: 'Wheat'}
        elif crop_types == 'Maize': dict_crop_types = {2: 'Maize'}
        else: raise ValueError('crop_types?')
        crop_ids = list(dict_crop_types.keys())
        crop_ids.sort()
        monthly_mask_names = ['{:02}.{}'.format(crop_id, dict_crop_types[crop_id]) for crop_id in crop_ids]
else:
    monthly_mask_names = [None]  # keep this list empty for non-monthly cases!!

bm = Basemap(projection='cyl', llcrnrlat=-56.5, urcrnrlat=84.5, llcrnrlon=-180., urcrnrlon=180., resolution='l')


# ----------------------------------------------------------------------------------------------------------------------
def read_change(variable, scale, scenario, gcm, ghm, period, monthly_mask_name=None):

    proxy = 'period_total_drought_months'
    if DUMMY:
        return np.random.rand(ny, nx)

    else:  # src.shape (ny, nx, ntime)  ntime = 2868 = 12*239
        if variable == 'pr':
            filename = f'{gcm}_hist{scenario}_pr_monthly_1861_2099_spi_{distribution}_{scale:02}_{drought_severity}_{proxy}.nc'
        else:
            soc_hist, soc_future = dict_soc[ghm]
            filename = f'{ghm}_{gcm}_hist{scenario}_{soc_hist}_{soc_future}_co2_{variable}_monthly_1861_2099_spi_{distribution}_{scale:02}_{drought_severity}_{proxy}.nc'
        if MONTHLY:
            filename = filename[:-3]+f'_{monthly_mask_name}.nc'
        srcpath = genpath(data_directory_main, variable, filename)
        try:
            srcs = Dataset(srcpath)[proxy][:]  # (nperiod, ny, nx)
            srcs = np.ma.masked_greater(srcs, 1e+19)  # just for the case...
            srcs = srcs[_periods.index(period)] - srcs[0]
            print(f'read: {period} {srcpath} {srcs.shape}')
            return srcs  # (ny, nx)
        except:
            #print('????: {} is NOT exist... Check!! Caution!!!!!!'.format(srcpath))
            #return np.full((ny, nx), 1.e+20)
            raise FileNotFoundError(f'{srcpath} is NOT exist...? Something is wrong. Check!!')


# ----------------------------------------------------------------------------------------------------------------------
def load_reuse_source(target, period, monthly_mask_name):
    if monthly_mask_name is None:
        filename = f'{target}_{period}_{distribution}_{drought_severity}.nc'
    else:
        #filename = '{}_{}_{}_{}_{}.nc'.format(target, period, monthly_mask_name, distribution, drought_severity)
        filename = f'{target}_{period}_{monthly_mask_name}_{distribution}_{drought_severity}.nc'
    input_directory = genpath(figure_directory_main, drought_severity, distribution, factors_directory, scale_directory, date_for_reuse)
    inputpath = genpath(input_directory, filename)
    src = Dataset(inputpath)[target][0]
    print(f'REUSE: {inputpath}')
    return src


# ----------------------------------------------------------------------------------------------------------------------
def write_to_ncfile(src, target, period, monthly_mask_name):

    baseyear = 1661
    complevel = 5
    lats = np.arange(89.75, -90, -0.5)
    lons = np.arange(-179.75, 180, 0.5)
    Times = [(date(dict_period[period][0],1,1)-date(baseyear,1,1)).days]
    nT = len(Times)

    # open a netcdf and write-in
    if monthly_mask_name is None:
        filename = f'{target}_{period}_{distribution}_{drought_severity}.nc'
    else:
        filename = f'{target}_{period}_{monthly_mask_name}_{distribution}_{drought_severity}.nc'
    #output_directory = genpath(figure_directory_main, drought_severity, distribution, scale_directory, today)
    #output_directory = genpath(figure_directory_main, drought_severity, distribution, factors_directory, scale_directory, today)
    #if not os.path.isdir(output_directory): os.makedirs(output_directory)
    outputpath = genpath(output_directory, filename)

    rootgrp = Dataset(outputpath, 'w', format='NETCDF4')

    rootgrp.description = 'ISIMIP2b drought propagetion analysis'
    import time
    rootgrp.history     = 'Created ' + time.ctime(time.time())
    rootgrp.source      = 'ISIMIP2b'
    rootgrp.title       = target
    rootgrp.institution = 'NIES'
    rootgrp.contact     = 'satoh.yusuke@nies.go.jp'
    rootgrp.version     = version

    time = rootgrp.createDimension('time', nT)
    lon  = rootgrp.createDimension('lon', nx)
    lat  = rootgrp.createDimension('lat', ny)

    times                    = rootgrp.createVariable('time','f8',('time',), zlib=True, complevel=complevel)
    times.units              = f'days since {baseyear}-01-01 00:00:00'

    longitudes               = rootgrp.createVariable('lon', 'f8',('lon',), zlib=True, complevel=complevel)
    longitudes.long_name     = 'longitude'
    longitudes.units         = 'degrees east'
    longitudes.standard_name = 'longitude'
    longitudes.axis          = 'X'

    latitudes                = rootgrp.createVariable('lat', 'f8',('lat',), zlib=True, complevel=complevel)
    latitudes.long_name      = 'latitude'
    latitudes.units          = 'degrees north'
    latitudes.standard_name  = 'latitude'
    latitudes.axis           = 'Y'

    srcs                     = rootgrp.createVariable(target, 'f4', ('time','lat','lon'),
                                                      zlib=True, complevel=complevel,
                                                      fill_value=1.e+20,
                                                      chunksizes=(1, ny, nx)
                                                      )
    #srcs.long_name           = longname  # longName
    srcs.missing_value       = np.float32(1.e+20)
    srcs.memo                = 'ID: ' + ', '.join([f'{i}: {factor}' for i, factor in enumerate(factors)])
    srcs.memo2               = 'drought type: ' + ', '.join(variables)
    srcs.memo3               = 'drought scale'  + ', '.join(str(scales))
    srcs.memo4               = 'scenario    : ' + ', '.join(scenarios)
    srcs.memo5               = 'gcm         : ' + ', '.join(gcms)
    srcs.memo6               = 'ghm         : ' + ', '.join(ghms)

    times[:]      = Times
    latitudes[:]  = lats
    longitudes[:] = lons
    srcs[:]       = src

    rootgrp.close()
    print(f'\nFinished writting   : {outputpath} {src.shape} {src.min()}-{src.max()}\n')


# ----------------------------------------------------------------------------------------------------------------------
def calc_residual_squaresum(srcs):

    squaresum = []
    for idrought, iscenario, igcm, ighm in itertools.product(range(ntype), range(nscenario), range(ngcm), range(nghm)):
        src = srcs[idrought, iscenario, igcm, ighm]  # (9)
        mean_group = src.mean()
        _squaresum = np.power(src - mean_group, 2).sum()
        squaresum.append(_squaresum)
    return sum(squaresum)


# ----------------------------------------------------------------------------------------------------------------------
def drawmap(src, topic, whatisthisabout, period, monthly_mask_name, vmin=None, vmax=None):

    src = np.ma.masked_equal(src, 1e+20)

    if 'unbiasedvariance' in topic:
        cmap = cm.cubehelix_r
    elif topic == 'frac-variance2median' or topic == 'frac-std2median':
        cmap = cm.RdBu_r
    else:
        cmap = cm.CMRmap_r

    if 'unbiasedvariance' in topic or topic == 'frac-variance2median' or topic == 'frac-std2median':
        fig1 = plt.figure(num=1, figsize=(9.9, 4.6))
        gs = gridspec.GridSpec(1, 1)  # (rows,cols)
        gs.update(left=0.01, right=0.93, bottom=0.05, top=0.95, hspace=0.02, wspace=0.01)
    else:
        fig1 = plt.figure(num=1, figsize=(8, 4.6))
        gs = gridspec.GridSpec(1, 1)  # (rows,cols)
        gs.update(left=0.01, right=0.99, bottom=0.05, top=0.95, hspace=0.02, wspace=0.01)

    # draw a map
    ax1 = plt.subplot(gs[0,0])
    ax1.axis('off')
    ax_pos = ax1.get_position()
    ax1.set_title(f'{whatisthisabout}', fontsize=20)

    im = bm.imshow(flipud(src[11:293]), vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
    bm.drawcoastlines(linewidth=0.2)

    if 'unbiasedvariance' in topic or topic == 'frac-variance2median' or topic == 'frac-std2median':
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax1)
        cax = divider.append_axes('right', '2%', pad='1%')
        cb = fig1.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=18)
    else:
        dx, dy, box_width, box_hight = 0.1, -0.02, 0.8, 0.06
        cax = fig1.add_axes([ax_pos.x0+dx, ax_pos.y0+dy+box_hight, box_width, box_hight])
        cb = plt.colorbar(im, cax=cax, pad=0.08, aspect=53, orientation='horizontal')
        cb.ax.tick_params(labelsize=30, direction='in')

    figurename = f'{topic}_{whatisthisabout}_{period}_{distribution}_{drought_severity}'
    if monthly_mask_name is not None: figurename = figurename+'_{}'.format(monthly_mask_name)
    #figure_directory = genpath(figure_directory_main, drought_severity, distribution, scale_directory, today)
    #figure_directory = genpath(figure_directory_main, drought_severity, distribution, factors_directory, scale_directory, today)
    #if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
    figurepath = genpath(figure_directory, figurename)
    for suffix in suffixes:
        plt.savefig(figurepath+suffix)
        print('save: {}'.format(figurepath+suffix))
    plt.close(1)


# ----------------------------------------------------------------------------------------------------------------------
def drawmap_dominantUS(src, whatisthisabout, terms, period, monthly_mask_name):

    src = np.ma.masked_equal(src, 1e+20)

    if len(terms) == 4:
        bounds = [0., 1., 2., 3., 4.]
        colors = np.divide([[51., 51., 255.], [191., 0., 6.], [ 169., 92., 164.], [255., 126., 51.]], 255.)
        dx, dy, box_width, box_hight = 0.45, 0.07, 0.5, 0.03
        tickposition = [0.5,1.5,2.5,3.5]
        if   factors == ['def', 'scn', 'gcm', 'ghm']: ticklabels = ['definition', 'scenario', 'gcm', 'gwm']
        elif factors == ['typ', 'scl', 'scn', 'mdl']: ticklabels = ['category', 'scale', 'scenario', 'model']
    ##elif len(terms) == 5:
    ##    bounds = [0., 1., 2., 3., 4., 5.]
    ##    colors = np.divide([[51., 51., 255.], [191., 0., 6.], [255., 126., 51.], [202., 158., 199.], [149., 207., 149.]], 255.)
    ##    #dx, dy, box_width, box_hight = 0.25, 0.03, 0.6, 0.03
    ##    dx, dy, box_width, box_hight = 0.25, 0.07, 0.6, 0.03
    ##    tickposition = [0.5,1.5,2.5,3.5,4.5]
    ##    ticklabels = ['definition', 'scenario', 'gcm', 'gwm', 'definition\ngwm', 'definition\ngcm\ngwm', 'all']
    ##elif len(terms) == 7:
    ##    bounds = [0., 1., 2., 3., 4., 5., 6., 7.]
    ##    colors = np.divide([[51., 51., 255.], [191., 0., 6.], [255., 126., 51.], [202., 158., 199.], [149., 207., 149.],[189., 227., 255.],[209., 255., 209.]], 255.)
    ##    #dx, dy, box_width, box_hight = 0.25, 0.03, 0.6, 0.03
    ##    dx, dy, box_width, box_hight = 0.25, 0.07, 0.6, 0.03
    ##    tickposition = [0.5,1.5,2.5,3.5,4.5,5.5,6.5]
    ##    ticklabels = ['definition', 'scenario', 'gcm', 'gwm', 'definition\ngwm', 'definition\ngcm\ngwm', 'all']
    else:
        raise ValueError(f'The size of terms is not 4: {terms}')

    # --- draw a map
    fig1 = plt.figure(num=2, figsize=(8, 4.6))
    gs = gridspec.GridSpec(1, 1)  # (rows,cols)
    gs.update(left=0.01, right=0.99, bottom=0.05, top=0.95, hspace=0.02, wspace=0.01)

    ax1 = plt.subplot(gs[0,0])
    plt.sca(ax1)
    ax1.axis('off')
    ax_pos = ax1.get_position()
    ax1.set_title(whatisthisabout.replace('_', ' '), fontsize=7.5)

    cmap = mpl.colors.ListedColormap(colors)
    ims = bm.imshow(flipud(src[11:293]), cmap=cmap, vmin=bounds[0], vmax=bounds[-1], interpolation='nearest')
    bm.drawcoastlines(linewidth=0.2)

    cax = fig1.add_axes([ax_pos.x0+dx, ax_pos.y0+dy+box_hight, box_width, box_hight])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, spacing='uniform', orientation='horizontal')
    cb.set_ticks(tickposition)
    cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=13, direction='in')

    figurename = f'{whatisthisabout}_{period}_{distribution}_{drought_severity}'
    if monthly_mask_name is not None: figurename = figurename+'_{}'.format(monthly_mask_name)
    #figure_directory = genpath(figure_directory_main, drought_severity, distribution, factors_directory, scale_directory, today)
    #if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
    figurepath = genpath(figure_directory, figurename)

    for suffix in suffixes:
        plt.savefig(figurepath+suffix)
        print('save: {}'.format(figurepath+suffix))
    plt.close(2)


# ----------------------------------------------------------------------------------------------------------------------
def find_dominant_uncertainty_source(src, topic_mask):

    YY, XX = np.where(~topic_mask)  # inverse the mask and pick up grids that is not masked = target grids
    dominant = np.full((ny, nx), 1e+20)
    for iy, ix in zip(YY, XX):
        if src[:,iy,ix].min() >= 1e+19:
            pass
        else:
            dominant[iy, ix] = src[:, iy, ix].argmax()
    dominant = np.ma.masked_equal(dominant, 1e+20)
    dominant = np.ma.masked_array(dominant, mask=seamask)
    dominant = np.ma.masked_array(dominant, mask=topic_mask)
    return dominant


def find_dominant_uncertainty_source_accumulated(src, factors_all, topic_mask):

    _src = []  # will be accumulated facts
    for ifactor, main_factor in enumerate(factors):  # ['def', 'scn', 'gcm', 'ghm']
        indices = []
        for ifctrs, fctrs in enumerate(factors_all):
            if main_factor in fctrs: indices.append(ifctrs)  # for 'def': [0, 4, 5, 6, 10, 11, 12, 14]
        # sum up variance_frac
        __src = np.zeros((ny, nx), 'float32')
        for i in indices:
            __src = __src + src[i]
        _src.append(__src)
    _src = np.array(_src)

    YY, XX = np.where(~topic_mask)  # inverse the mask and pick up grids that is not masked = target grids
    dominant = np.full((ny, nx), 1e+20)
    for iy, ix in zip(YY, XX):
        if src[:,iy,ix].min() >= 1e+19:
            pass
        else:
            dominant[iy, ix] = _src[:, iy, ix].argmax()
    dominant = np.ma.masked_equal(dominant, 1e+20)
    dominant = np.ma.masked_array(dominant, mask=seamask)
    dominant = np.ma.masked_array(dominant, mask=topic_mask)
    return dominant


# ----------------------------------------------------------------------------------------------------------------------
def calc_statistics(src, level, *targets, igrid=None):  # src: (ndefinition, nscenario, ngcm, nghm)

    def extract_and_mean(_src, *targets_and_indices, CHECK=False):
        # --- get target axes front
        head = []
        tail = list(range(len(factors)))
        for (_target, _index) in targets_and_indices:
            target_axis = factors.index(_target)
            head.append(target_axis)
            tail.remove(target_axis)
        axes = tuple(head + tail)
        _src = np.transpose(_src, axes=axes)
        # --- excract from the lower dimension  (= higher axis)
        for (_target, _index) in targets_and_indices:
            _src = _src[_index]
        # --- check
        if igrid == 0 and CHECK:
            print('   extract_and_mean for {}. transpose to {}, extract {} and mean'.format(targets_and_indices, axes, _src.shape))
        return _src.mean()

    dict_group_items = {}
    dict_member_items = dict_n_factormembers.copy()
    axes_member = list(range(len(factors)))
    for target in targets:
        dict_group_items[target] = dict_n_factormembers[target]
        del dict_member_items[target]
        axes_member.remove(factors.index(target))
    axes_member = tuple(axes_member)  # list to tuple
    ngroup = np.prod(list(dict_group_items.values()))
    nmembers = np.prod(list(dict_member_items.values()))
    axes_group = list(range(len(factors)))
    for axis in axes_member:
        axes_group.remove(axis)
    if igrid == 0:
        print('\n==================\nlevel: {}, targets: {}, ngroup: axes={} {}<-{}, nmembers: axes={} {}<-{}\n'.format(level, targets,
                                                                                                axes_group, ngroup, dict_group_items,
                                                                                                axes_member, nmembers, dict_member_items))

    mean_overall = src.mean()  # the overall grand mean

    if anova_level == 2:
        if level == 'main':  # single factor
            if targets[0] == factors[0] and igrid % 5000 == 0:
                print('\nsrc {} @{}:\n{}'.format(src.shape, igrid, src.tolist()))
            powered_diffs = []
            for i in range(dict_n_factormembers[targets[0]]):
                CHECK = True if i == 0 else False
                if igrid == 0 and CHECK: print('--- main factors')
                mean_group = extract_and_mean(src, (targets[0], i), CHECK=CHECK)
                powered_diff = np.power(mean_group - mean_overall, 2)
                powered_diffs.append(powered_diff)
            squaresum = nmembers * np.sum(powered_diffs)
            if igrid % 5000 == 0:
                print('squaresum (main factor):\n{}\n'.format(squaresum))
        elif level == '1st_inter':  # interaction between two factors
            powered_diffs = []
            for i, j in itertools.product(range(dict_n_factormembers[targets[0]]), range(dict_n_factormembers[targets[1]])):
                CHECK = True if i+j == 0 else False
                if igrid == 0 and CHECK:
                    print('--- 1st interactions')
                mean_group      = extract_and_mean(src, (targets[0], i), (targets[1], j), CHECK=CHECK)
                mean_subgroup_i = extract_and_mean(src, (targets[0], i), CHECK=CHECK)
                mean_subgroup_j = extract_and_mean(src, (targets[1], j), CHECK=CHECK)
                powered_diff = np.power(mean_group - mean_subgroup_i - mean_subgroup_j + mean_overall, 2)
                powered_diffs.append(powered_diff)
            squaresum = nmembers * np.sum(powered_diffs)
            if igrid % 5000 == 0:
                print('squaresum (main factor):\n{}\n'.format(squaresum))
        elif level == '2nd_inter':  # interaction among three factors
            powered_diffs = []
            for i, j, k in itertools.product(range(dict_n_factormembers[targets[0]]), range(dict_n_factormembers[targets[1]]), range(dict_n_factormembers[targets[2]])):
                CHECK = True if i+j+k == 0 else False
                if igrid == 0 and CHECK:
                    print('--- 2nd interactions')
                mean_group         = extract_and_mean(src, (targets[0], i), (targets[1], j), (targets[2], k), CHECK=CHECK)
                mean_subgroup_ij   = extract_and_mean(src, (targets[0], i), (targets[1], j), CHECK=CHECK)
                mean_subgroup_ik   = extract_and_mean(src, (targets[0], i), (targets[2], k), CHECK=CHECK)
                mean_subgroup_jk   = extract_and_mean(src, (targets[1], j), (targets[2], k), CHECK=CHECK)
                mean_subsubgroup_i = extract_and_mean(src, (targets[0], i), CHECK=CHECK)
                mean_subsubgroup_j = extract_and_mean(src, (targets[1], j), CHECK=CHECK)
                mean_subsubgroup_k = extract_and_mean(src, (targets[2], k), CHECK=CHECK)
                powered_diff = np.power(mean_group - mean_subgroup_ij - mean_subgroup_ik - mean_subgroup_jk
                                                   + mean_subsubgroup_i + mean_subsubgroup_j + mean_subsubgroup_k - mean_overall, 2)
                powered_diffs.append(powered_diff)
            squaresum = nmembers * np.sum(powered_diffs)
            if igrid % 5000 == 0:
                print('squaresum (main factor):\n{}\n'.format(squaresum))
        #elif level == '3rd_inter':  # interaction among four factors

        return squaresum
    
    else:
        raise ValueError('check anova_level...')

    """
        if level == 'main':  # single factor
            mean_group = src.mean(axis=axes_member)
            squaresum = nmembers * np.power(mean_group - mean_overall, 2).sum()
    if anova_level == 1:  # DO NOT USE THIS!! This may need to be updated!!
        dof = ngroup - 1  # degree of freedom
        mean_group = src.mean(axis=axes_member)
        squaresum = np.power(mean_group - mean_overall, 2).sum()
        meansquare = squaresum / dof  # <- unbiased variance
        return meansquare

    elif anova_level == 3:  # DO NOT USE THIS!! This may need to be updated!!
        dof = ngroup - 1  # degree of freedom
        mean_group = src.mean(axis=axes_member)
        squaresum = np.power(mean_group - mean_overall, 2).sum() * nmembers
        meansquare = squaresum / dof  # <- variance
        return meansquare
    """


# ----------------------------------------------------------------------------------------------------------------------
def anova_level2(srcs, period, monthly_mask_name, YY, XX):

    if REUSE:
        SST = load_reuse_source('totalsumsquareddeviation_overall', period, monthly_mask_name)
        SS_main = np.array([load_reuse_source(f'SS0main_{}'.format(target), period, monthly_mask_name) for target in factors])
        SS_1st_interaction = np.array([load_reuse_source('SS1stinteraction_{}+{}'.format(*target), period, monthly_mask_name) for target in factor2combinations])
        SS_2nd_interaction = np.array([load_reuse_source('SS2ndinteraction_{}+{}+{}'.format(*target), period, monthly_mask_name) for target in factor3combinations])
        SS_residual = load_reuse_source('SS3rdinteraction_full', period, monthly_mask_name)
        SS_main = np.ma.masked_equal(SS_main, 1e+20)
        SS_1st_interaction = np.ma.masked_equal(SS_1st_interaction, 1e+20)
        SS_2nd_interaction = np.ma.masked_equal(SS_2nd_interaction, 1e+20)
        topic_mask = SST.mask
    else:
        SST = np.power(srcs - srcs.mean(), 2).sum(axis=(0,1,2,3))  # (ny, nx)
        topic_mask = SST.mask
    
        # set empty field...
        SS_main            = np.full((len(factors), ny, nx), 1e+20)
        SS_1st_interaction = np.full((len(factor2combinations), ny, nx), 1e+20)
        SS_2nd_interaction = np.full((len(factor3combinations), ny, nx), 1e+20)
        #SS_3rd_interaction = np.full((ny, nx), 1e+20)
        #  for each grid...
        for igrid, (iy, ix) in enumerate(zip(YY, XX)):
            if igrid % 50 == 0:
                print ('igrid = {}'.format(igrid))
    
            src = srcs[..., iy, ix]  # (ntype, nscenario, ngcm, nghm)  <- Caution!! This order has to be consistent with "factors".   Mask is all True.
            for i, target in enumerate(factors):
                SS_main[i, iy, ix] = calc_statistics(src, 'main', target, igrid=igrid)  # unbiased variabnce
            for i, (target1, target2) in enumerate(factor2combinations):
                SS_1st_interaction[i, iy, ix] = calc_statistics(src, '1st_inter', target1, target2, igrid=igrid)
            for i, (target1, target2, target3) in enumerate(factor3combinations):
                SS_2nd_interaction[i, iy, ix] = calc_statistics(src, '2nd_inter', target1, target2, target3, igrid=igrid)
            #SS_3rd_interaction[iy, ix] = calc_statistics(src, '3rd_inter', *factors, igrid=igrid)

        """
        # multiprocessing...  # TODO
        """
		
        # --- reproduce the topic mask...
        SS_main = np.ma.masked_equal(SS_main, 1e+20)                          # (4, ny, nx)
        SS_1st_interaction = np.ma.masked_equal(SS_1st_interaction, 1e+20)    # (6, ny, nx)
        SS_2nd_interaction = np.ma.masked_equal(SS_2nd_interaction, 1e+20)    # (4, ny, nx)
        # --- calc the 3rd order interaction term (= residual)
        SS_residual = SST \
                      - np.ma.masked_greater(SS_main.sum(axis=0),            1e+19).filled(0) \
                      - np.ma.masked_greater(SS_1st_interaction.sum(axis=0), 1e+19).filled(0) \
                      - np.ma.masked_greater(SS_2nd_interaction.sum(axis=0), 1e+19).filled(0) \
                      #- np.ma.masked_greater(SS_3rd_interaction, 1e+19).filled(0)  # the residual
        SS_residual = np.ma.masked_array(SS_residual, mask=np.resize(topic_mask, SS_residual.shape))  # (ny, nx)

    # === post-process =============================================
    # --- calc fraction
    variance_frac_main     = np.divide(SS_main,            SST)
    variance_frac_1st      = np.divide(SS_1st_interaction, SST)
    variance_frac_2nd      = np.divide(SS_2nd_interaction, SST)
    variance_frac_residual = np.divide(SS_residual,        SST)
    # --- remove nan
    variance_frac_main[np.isnan(variance_frac_main)]         = 0
    variance_frac_1st[np.isnan(variance_frac_1st)]           = 0
    variance_frac_2nd[np.isnan(variance_frac_2nd)]           = 0
    variance_frac_residual[np.isnan(variance_frac_residual)] = 0
    # --- mask sea grids
    variance_frac_main = np.ma.masked_array(variance_frac_main, mask=np.resize(seamask, variance_frac_main.shape))              # (4, ny, nx)
    variance_frac_1st = np.ma.masked_array(variance_frac_1st, mask=np.resize(seamask, variance_frac_1st.shape))                 # (6, ny, nx)
    variance_frac_2nd = np.ma.masked_array(variance_frac_2nd, mask=np.resize(seamask, variance_frac_2nd.shape))                 # (4, ny, nx)
    variance_frac_residual = np.ma.masked_array(variance_frac_residual, mask=np.resize(seamask, variance_frac_residual.shape))  # (   ny, nx)

    # --- dominant uncertainty source among...
    # main factors
    dominant_source_main = find_dominant_uncertainty_source(variance_frac_main, topic_mask)  # (ny, nx)
    """
    # main + 'def+ghm'
    variance_fracs = np.concatenate([variance_frac_main,
                                     variance_frac_1st[2].reshape(-1, ny, nx)], axis=0)
    selected_5_terms = factors + [factor2combinations[2]]
    dominant_source_selected_5 = find_dominant_uncertainty_source(variance_fracs, topic_mask)  # (ny, nx)
    # main + 'def+gcm' + 'def+ghm' + 'all'
    variance_fracs = np.concatenate([variance_frac_main,
                                     variance_frac_1st[2].reshape(-1, ny, nx),
                                     variance_frac_2nd[2].reshape(-1, ny, nx),
                                     variance_frac_residual.reshape(-1,ny,nx)], axis=0)
    selected_7_terms = factors + [factor2combinations[2]] + [factor3combinations[2]] + ['all']
    dominant_source_selected_7 = find_dominant_uncertainty_source(variance_fracs, topic_mask)  # (ny, nx)
    """ 
    # dominant based on accumulated fractions
    variance_fracs = np.concatenate([variance_frac_main,  # for [def, scn, gcm, ghm]
                                     variance_frac_1st,   # for [('def', 'scn'), ('def', 'gcm'), ('def', 'ghm'), ('scn', 'gcm'), ('scn', 'ghm'), ('gcm', 'ghm')]
                                     variance_frac_2nd,   # for [('def', 'scn', 'gcm'), ('def', 'scn', 'ghm'), ('def', 'gcm', 'ghm'), ('scn', 'gcm', 'ghm')]
                                     variance_frac_residual.reshape(-1,ny,nx)],  # for [('def', 'scn', 'gcm')]
                                     axis=0)
    factors_all = [(f,) for f in factors] + factor2combinations + factor3combinations + [(factors[0], factors[1], factors[2], factors[3])]
    dominant_source_accumulated = find_dominant_uncertainty_source_accumulated(variance_fracs, factors_all, topic_mask)  # (ny, nx)

    # drawmaps + write out to netcdf
    drawmap(SST, 'totalsumsquareddeviation', 'overall', period, monthly_mask_name)
    write_to_ncfile(SST, 'totalsumsquareddeviation_overall', period, monthly_mask_name)

    nsample = ntype*nscale*nscenario*ngcm*nghm-1
    unbiasedvariance_overall = np.sqrt(np.divide(SST, nsample))
    if   climatology == '30yrs': uv_vmax1 = 200
    elif climatology == '50yrs': uv_vmax1 = 220 
    drawmap(unbiasedvariance_overall, 'unbiasedvariance', 'overall', period, monthly_mask_name, vmax=uv_vmax1)
    write_to_ncfile(unbiasedvariance_overall, 'unbiasedvariance_overall', period, monthly_mask_name)
    for i, target in enumerate(factors):
        unbiasedvariance = np.sqrt(np.divide(SS_main[i], dict_n_factormembers[target]-1))
        drawmap(unbiasedvariance, 'unbiasedvariance_mainfactor', target, period, monthly_mask_name)
        write_to_ncfile(unbiasedvariance, f'unbiasedvariance_mainfactor_{target}', period, monthly_mask_name)
    #if srcs is not None:
    overall_median = np.median(srcs, axis=(0,1,2,3))
    variance_to_median = np.divide(unbiasedvariance_overall, overall_median)
    std_to_median = np.divide(np.sqrt(unbiasedvariance_overall), overall_median)  # coefficient of variation
    drawmap(overall_median, 'median', 'overall', period, monthly_mask_name)
    drawmap(variance_to_median, 'frac-variance2median', 'overall', period, monthly_mask_name, vmin=-30, vmax=30)
    drawmap(std_to_median, 'frac-std2median', 'overall', period, monthly_mask_name, vmin=-1.4, vmax=1.4)
    write_to_ncfile(overall_median, 'median_overall', period, monthly_mask_name)
    write_to_ncfile(variance_to_median, 'frac-variance2median_overall', period, monthly_mask_name)
    write_to_ncfile(std_to_median, 'frac-std2median_overall', period, monthly_mask_name)
    # ---
    for i, target in enumerate(factors):
        drawmap(SS_main[i], 'SS0main', target, period, monthly_mask_name)
        write_to_ncfile(SS_main[i], 'SS0main_{}'.format(target), period, monthly_mask_name)
        drawmap(variance_frac_main[i], 'frac_0th_main', target, period, monthly_mask_name, vmin=0, vmax=1.0)
        write_to_ncfile(variance_frac_main[i], 'frac_0th_main_{}'.format(target), period, monthly_mask_name)
    for i, target in enumerate(factor2combinations):
        drawmap(SS_1st_interaction[i], 'SS1stinteraction', '{}+{}'.format(*target), period, monthly_mask_name)
        write_to_ncfile(SS_1st_interaction[i], 'SS1stinteraction_{}+{}'.format(*target), period, monthly_mask_name)
        drawmap(variance_frac_1st[i], 'frac_1st_interaction', '{}+{}'.format(*target), period, monthly_mask_name, vmin=0, vmax=1.0)
        write_to_ncfile(variance_frac_1st[i], 'frac_1st_interaction_{}+{}'.format(*target), period, monthly_mask_name)
    for i, target in enumerate(factor3combinations):
        drawmap(SS_2nd_interaction[i], 'SS2ndinteraction', '{}+{}+{}'.format(*target), period, monthly_mask_name)
        write_to_ncfile(SS_2nd_interaction[i], 'SS2ndinteraction_{}+{}+{}'.format(*target), period, monthly_mask_name)
        drawmap(variance_frac_2nd[i], 'frac_2nd_interaction', '{}+{}+{}'.format(*target), period, monthly_mask_name, vmin=0, vmax=1.0)
        write_to_ncfile(variance_frac_2nd[i], 'frac_2nd_interaction_{}+{}+{}'.format(*target), period, monthly_mask_name)
    #drawmap(variance_frac_3rd, 'variance_fraction_3rd_interaction', 'full', period, monthly_mask_name, vmin=0, vmax=1.0)
    #write_to_ncfile(variance_frac_3rd, 'frac_3rd.interaction_full', period, monthly_mask_name)
    drawmap(SS_residual, 'SS3rdinteraction', 'full', period, monthly_mask_name)
    write_to_ncfile(SS_residual, 'SS3rdinteraction_full', period, monthly_mask_name)
    drawmap(variance_frac_residual, 'frac_residual', 'full', period, monthly_mask_name)
    write_to_ncfile(variance_frac_residual, 'frac_residual_full', period, monthly_mask_name)

    drawmap_dominantUS(dominant_source_main,        'dominantsource_main',        factors,          period, monthly_mask_name)
    ##drawmap_dominantUS(dominant_source_selected_5,  'dominantsource_selected_5',  selected_5_terms, period, monthly_mask_name)
    ##drawmap_dominantUS(dominant_source_selected_7,  'dominantsource_selected_7',  selected_7_terms, period, monthly_mask_name)
    drawmap_dominantUS(dominant_source_accumulated, 'dominantsource_accumulated', factors,          period, monthly_mask_name)
    write_to_ncfile(dominant_source_main,        'dominantsource_main',        period, monthly_mask_name)
    ##write_to_ncfile(dominant_source_selected_5,  'dominantsource_selected_5',  period, monthly_mask_name)
    ##write_to_ncfile(dominant_source_selected_7,  'dominantsource_selected_7',  period, monthly_mask_name)
    write_to_ncfile(dominant_source_accumulated, 'dominantsource_accumulated', period, monthly_mask_name)

    SS_residual_main = SST - np.ma.masked_greater(SS_main.sum(axis=0), 1e+19).filled(0)
    variance_frac_residual_main = np.ma.masked_array(np.divide(SS_residual_main,        SST), mask=np.resize(seamask, SS_residual_main.shape))
    drawmap(variance_frac_residual_main, 'frac_residualmain', 'residual_from_main', period, monthly_mask_name, vmin=0, vmax=1.0)
    write_to_ncfile(variance_frac_residual_main, 'frac_residualmain_full', period, monthly_mask_name)

"""
def anova_level1(srcs, period, monthly_mask_name, YY, XX):

    #  set empty field...
    variances = np.full((4, ny, nx), 1e+20)
    #  for each grid...
    for iy, ix in zip(YY, XX):
        src = srcs[..., iy, ix]  # (ntype, nscenario, ngcm, nghm)
        for i, target in enumerate(factors):
            variances[i, iy, ix] = calc_statistics(target, src)  # unbiased variabnce
    # post-process
    variance_rates = np.divide(variances, variances.sum(axis=0))
    variance_rates = np.ma.masked_array(variance_rates, mask=np.resize(seamask, variance_rates.shape))
    dominant_source = find_dominant_uncertainty_source(variances)  # (ny, nx)
    # drawmaps
    for i, target in enumerate(factors):
        drawmap(variances[i], 'variance', target, period)
        drawmap(variance_rates[i], 'variance_rate', target, period)
    drawmap_dominantUS(dominant_source, 'dominant_uncertainty_source', period)


def anova_level3(srcs, period, monthly_mask_name, YY, XX):

    # set empty field...
    residual_meansquare_map = np.full((ny, nx), 1e+20)
    meansquare_map = np.full((4, ny, nx), 1e+20)
    f_values = np.full((4, ny, nx), 1e+20)
    #  for each grid...
    for iy, ix in zip(YY, XX):
        #src = srcs[..., iy-1:iy+1+1, ix-1:ix+1+1]  # (ntype, nscenario, ngcm, nghm, 3, 3)
        #src = src.reshape(src.shape[:-2]+(-1,))
        #shp = src.shape                            # (ntype, nscenario, ngcm, nghm, 9)
        #residual_dof = ntype * nscenario * ngcm * nghm * (9-1)
        src = srcs[..., iy, ix]  # (ntype, nscenario, ngcm, nghm)
        residual_dof = ntype * nscenario * ngcm * nghm - 1  # TODO!!!!! how should I set dof here??? Maybe, the reference is wrong?
        residual_squaresum = calc_residual_squaresum(src)
        residual_meansquare = residual_squaresum / residual_dof
        residual_meansquare_map[iy, ix] = residual_meansquare
        for i, target in enumerate(factors):
            meansquare = calc_statistics(target, src)
            meansquare_map[i, iy, ix] = meansquare
            f_values[i, iy, ix] = meansquare / residual_meansquare
    # post-process
    dominant_source = find_dominant_uncertainty_source(f_values)  # (ny, nx)
    # drawmaps
    drawmap(residual_meansquare_map, 'residual_meansquare', '', period)
    for i, target in enumerate(factors):
        drawmap(meansquare_map[i], 'meansquare', target, period)
        drawmap(f_values[i], 'Fvalue', target, period)
    drawmap_dominantUS(dominant_source, 'dominant_uncertainty_source', period)
"""


# ----------------------------------------------------------------------------------------------------------------------
def main():

    for period, monthly_mask_name in itertools.product(periods, monthly_mask_names):
        strTime = datetime.datetime.now()

        srcs = np.array([[[[[read_change(variable, scale, scenario, gcm, ghm, period, monthly_mask_name) for ghm in ghms]
                                                                                                         for gcm in gcms]
                                                                                                         for scenario in scenarios]
                                                                                                         for scale in scales]
                                                                                                         for variable in variables])  # (ntype,nscale,nscenario,ngcm,nghm,ny,nx)
        # 5s dimensions -> 4 dimensions
        if factors == ['def', 'scn', 'gcm', 'ghm']:
            srcs = srcs.reshape(-1, nscenario, ngcm, nghm, ny, nx)  # (ndefinition, nscenarios, ngcm, nghm, ny, nx)
        elif factors == ['typ', 'scl', 'scn', 'mdl']:
            srcs = srcs.reshape(ntype, nscale, nscenario, nmdl, ny, nx)  # (ndefinition, nscenarios, nmdl, ny, nx)
        srcs = np.ma.masked_greater(srcs, 1e+19)

        landgrid_flag = np.where(seamask, 0, 1)  # (ny, nx)
        YY, XX = np.where(landgrid_flag==1)
        if   anova_level == 1:
            anova_level1(srcs, period, monthly_mask_name, YY, XX)
        elif anova_level == 2:
            anova_level2(srcs, period, monthly_mask_name, YY, XX)
        elif anova_level == 3:
            anova_level3(srcs, period, monthly_mask_name, YY, XX)
        else:
            raise ValueError('check anova_level...')
        
        endTime = datetime.datetime.now()
        diffTime = endTime - strTime
        print('took {} min in total.'.format(int(diffTime.seconds/60)))

    print('This process was successfully DONE!!  d(^o^)b')


if __name__ == '__main__':
    main()
