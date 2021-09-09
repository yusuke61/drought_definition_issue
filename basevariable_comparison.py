#!/usr/bin/env python
# To compare uncertainties in base variable with that of main results  (for response to the reviewer)  >>> SupFig.2&3, SupFig.6&7
# By Yusuke Satoh
# On 2021/08/03
import os
import sys
import time
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import cm, gridspec
from mpl_toolkits.basemap import Basemap
from scipy import stats

# ----------------------------------------------------------------------------------------------------------------------
TEST = False
MAP = True
SCATTER = True

diff_type = sys.argv[1]
#diff_type = 'clim_diff'
#diff_type = 'clim_diff_pct'

stats_types = [sys.argv[2]]
#stats_types = ['median', 'std', 'iqr']

seasons = [sys.argv[3]]
#seasons = ['ANN', 'DJF', 'JJA']
#seasons = ['ANN']
#seasons = ['DJF', 'JJA']

topics = ['average', 'percentile_10', 'frequency']
variables = ['pr', 'qtot', 'soilmoist']
ghms = ['matsiro', 'cwatm', 'clm45', 'h08', 'jules-w1', 'lpjml', 'watergap2']
gcms = ['hadgem2-es','ipsl-cm5a-lr','gfdl-esm2m','miroc5']
scenarios = ['rcp26','rcp60','rcp85']
period = 'farfuture'
si_scale = 3  # standardized index's temporal scale

if TEST:
    variables = ['qtot', 'soilmoist']
    variables = ['soilmoist']
    ghms = ['matsiro']
    gcms = ['hadgem2-es', 'ipsl-cm5a-lr']
    scenarios = ['rcp85']
    seasons = ['ANN']


# basically, you don't need to edit here -------------------------------------------------------------------------------
syear, eyear = 1861, 2099
years = range(syear, eyear+1)
nyear = len(years)
dict_period = {
    'base_period': (1861, 1890),  # 30yrs
    'recent30yrs': (1990, 2019),  # 30yrs
    'nearfuture':  (2035, 2064),  # 30yrs
    'farfuture':   (2070, 2099),
    }
si_base,   ei_base   = years.index(dict_period['base_period'][0]), years.index(dict_period['base_period'][1])
si_future, ei_future = years.index(dict_period[period][0]),        years.index(dict_period[period][1])
periods = ['base_period', 'recent30yrs', 'nearfuture', 'farfuture']
ny, nx = 360, 720
suffixes = ['png', 'pdf']

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

data_directory_main = '/data/rg001/sgec0017/data/isimip2b.standardized_drought'
data_directory_main_in = os.path.join(data_directory_main, 'in')
data_directory_main_out = os.path.join(data_directory_main, 'climate_indices_postprocessed',
                                       'base_1861-1960', 'v2', 'climatology_30yrs')
figure_directory_main = '/data/rg001/sgec0017/data/figure_box/isimip2b.standardized_drought/basevariable_comparison'
if TEST: figure_directory_main = figure_directory_main + '.TEST'
if not os.path.isdir(figure_directory_main): os.makedirs(figure_directory_main)

landseamask_path = '/data/rg001/sgec0017/data/mapmask/ISIMIP2b_landseamask_generic.nc4'
landseamask = Dataset(landseamask_path)['LSM'][0].mask
greenlandmask_path = '/data/rg001/sgec0017/data/mapmask/GAUL/flt/gaul2014_05deg.flt'
greenlandmask = np.ma.masked_equal(np.fromfile(greenlandmask_path, 'float32').reshape(360,720),98).mask
landseamask = np.ma.mask_or(landseamask, greenlandmask)

projection = 'cyl'
resolution = 'l'
bm = Basemap(projection=projection,llcrnrlat=-56.5,urcrnrlat=84.5,llcrnrlon=-180.,urcrnrlon=180.,resolution=resolution)

dict_unit = {
    'pr'       : 'kg m-2 s-1',
    'qtot'     : 'kg m-2 s-1',
    'soilmoist': 'kg m-2',
    }
dict_varname = {
    'pr'       : 'pr',
    'qtot'     : 'ro',
    'soilmoist': 'sm',
    }
dict_standardizedname = {
    'pr'       : 'SPI3',
    'qtot'     : 'SRI3',
    'soilmoist': 'SSI3',
    }

# ----------------------------------------------------------------------------------------------------------------------
def load_input(topic, variable, ghm, gcm, scenario, season):

    start_time = time.time()
    soc_hist, soc_scenario = dict_soc[ghm]

    if topic == 'frequency':  # total number of drought month during the period (30yrs)
        if variable == 'pr':
            filename = f'{gcm}_hist{scenario}_{variable}_monthly_1861_2099_spi_gamma_{si_scale:02}_severe_period_total_drought_months.nc'
        else:
            filename = f'{ghm}_{gcm}_hist{scenario}_{soc_hist}_{soc_scenario}_co2_{variable}_monthly_1861_2099_spi_gamma_{si_scale:02}_severe_period_total_drought_months.nc'
        src_path = os.path.join(data_directory_main_out, variable, season, filename)
        ncvariablename = 'period_total_drought_months'
        with Dataset(src_path) as dataset:
            src = dataset.variables[ncvariablename][:]  # (ny, nx, nt)  nt=2868=12*239   or (4, ny, nx)
            print(f'load: {src_path} {src.shape}')
    else:  # for average and percentile_10 of base variables
        if variable == 'pr':  # [kg m-2 s-1]
            filename = f'{gcm}_hist{scenario}_{variable}_monthly_1861_2099.nc4'
        else:  # soilmosit [kg m-2], qtot [kg m-2 s-1]
            filename = f'{ghm}_{gcm}_hist{scenario}_{soc_hist}_{soc_scenario}_co2_{variable}_monthly_1861_2099.nc4'
        src_path = os.path.join(data_directory_main_in, variable, filename)
        ncvariablename = variable
        with Dataset(src_path) as dataset:
            src = dataset.variables[ncvariablename][:]  # (ny, nx, nt)  nt=2868=12*239   or (4, ny, nx)
            print(f'load: {src_path} {src.shape}')
        src = src.reshape(ny, nx, nyear, 12)           # (ny, nx, 239, 12)
        # seasonal average   (ny, nx, 239, 12) --> (ny, nx, 239)
        if season == 'ANN':
            src = src.mean(axis=3)
        elif season == 'JJA':
            src = src[:,:,:,5:8].mean(axis=3)
        elif season == 'DJF':
            src = np.concatenate((src[:,:,:,-1].reshape(ny,nx,nyear,1), src[:,:,:,:2]), axis=3).mean(axis=3)
        else:
            raise ValueError(f'season is {season}. Not ANN, JJA, nor DJF...')
    print(f'@ load_input:           elapsed_time --> {time.time()-start_time:.2f} [sec]')
    return src


# ----------------------------------------------------------------------------------------------------------------------
def gen_src(topic, variable, scenario, stats_type, season):

    start_time = time.time()
    def load_and_postprocess(topic, variable, ghm, gcm, scenario, season):
        _start_time = time.time()

        # load original src
        _src = load_input(topic, variable, ghm, gcm, scenario, season)  # (ny, nx, nyear) or (4, ny, nx)

        if diff_type == 'clim_diff':  # climatological difference  --> (ny,nx)
            if topic == 'frequency':
                _src = _src[periods.index(period)] - _src[periods.index('base_period')]  # (ny,nx)
            else:  # average, percentile_10
                if topic == 'average':
                    _src = np.mean(_src[:,:,si_future:ei_future], axis=2) - np.mean(_src[:,:,si_base:ei_base], axis=2)  # (ny,nx)
                elif topic == 'percentile_10':
                    _src = np.percentile(_src[:, :, si_future:ei_future], 10, axis=2) - np.percentile(_src[:, :, si_base:ei_base], 10, axis=2)  # (ny,nx)
        elif diff_type == 'clim_diff_pct':
            if topic == 'frequency':
                _src = np.divide(_src[periods.index(period)] - _src[periods.index('base_period')], 
                                 _src[periods.index('base_period')]
                                 )  # (ny,nx)
            else:  # average, percentile_10
                if topic == 'average':
                    _src = np.divide(np.mean(_src[:,:,si_future:ei_future], axis=2) - np.mean(_src[:,:,si_base:ei_base], axis=2), 
                                     np.mean(_src[:,:,si_base:ei_base], axis=2)
                                     )  # (ny,nx)
                elif topic == 'percentile_10':
                    _src = np.divide(np.percentile(_src[:, :, si_future:ei_future], 10, axis=2) - np.percentile(_src[:, :, si_base:ei_base], 10, axis=2),
                                     np.percentile(_src[:, :, si_base:ei_base], 10, axis=2)
                                     )  # (ny,nx)
            _src = _src * 100  # [%]
        else:
            raise ValueError(f'diff_type is {diff_type}. Not clim_diff nor clim_diff_pct')
        _src[np.isnan(_src)] = 1e+20
        _src[np.isinf(_src)] = 1e+20
        print(f'@ load_and_postprocess: elapsed_time --> {time.time()-_start_time:.2f} [sec]')
        return _src

    print(f'\n-----\n>>> {topic} {variable}')
    src = np.array([load_and_postprocess(topic, variable, ghm, gcm, scenario, season) for ghm in ghms for gcm in gcms])  # (nensemble, ny, nx)  
    src = np.ma.masked_greater(src, 1e+18)
    # ensemble stats
    if stats_type == 'median':
        src = np.nanmedian(src, axis=0)
    elif stats_type == 'std':
        src = np.nanstd(src, axis=0)
    elif stats_type == 'iqr':
        src = stats.iqr(src, axis=0, rng=(25, 75))
    else:
        raise ValueError(f'stats_type is {stats_type}. Not median, std nor iqr.')
    src[np.isnan(src)] = 1e+20
    src[np.isinf(src)] = 1e+20
    src = np.ma.masked_greater(src, 1e+18)
    print(f'@gen_src:               elapsed_time --> {time.time()-start_time:.2f} [sec]\n-----\n')
    print(f' >>> {src.shape}   {src.min()}-{src.max()}')
    return src


# ----------------------------------------------------------------------------------------------------------------------
def find_unit(topic, variable, season):

    if diff_type == 'clim_diff':
        if topic == 'frequency':
            if season == 'ANN':
                unit = 'months in 30 years'
            else:
                unit = f'months during {season}s in 30 years'
        else:
            unit = dict_unit[variable]
    elif diff_type == 'clim_diff_pct':
        unit = '%'
    return unit


# ----------------------------------------------------------------------------------------------------------------------
def draw_map(src, topic, variable, stats_type, scenario, season):

    print(f'--- @draw_map ({topic}): {variable}')

    src = np.ma.masked_array(src, mask=landseamask)[11:293]

    if diff_type == 'clim_diff':
        pctl = 95
    elif diff_type == 'clim_diff_pct':
        pctl = 90
    _max = np.percentile(np.abs(src.compressed()), pctl)
    # _max for exceptions
    if diff_type == 'clim_diff_pct' and stats_type == 'std':
        if variable == 'soilmoist' and season == 'ANN' and topic == 'frequency':  _max = 1400
        elif variable == 'qtot' and season == 'JJA' and topic == 'average':       _max = 210 
        elif variable == 'qtot' and season == 'JJA' and topic == 'percentile_10': _max = 810 
        elif variable == 'qtot' and season == 'JJA' and topic == 'frequency':     _max = 1150
        elif variable == 'qtot' and season == 'DJF' and topic == 'average':       _max = 1650
        elif variable == 'qtot' and season == 'DJF' and topic == 'percentile_10': _max = 1940 
        elif variable == 'qtot' and season == 'DJF' and topic == 'frequency':     _max = 850
        elif variable == 'qtot' and season == 'ANN' and topic == 'average':       _max = 190
        elif variable == 'qtot' and season == 'ANN' and topic == 'percentile_10': _max = 205
        elif variable == 'qtot' and season == 'ANN' and topic == 'frequency':     _max = 1610

    unit = find_unit(topic, variable, season)
    print(f'src  {src.min()}-{src.max()} ---> _max={_max}')

    fig = plt.figure(figsize=(4,2.4))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.22, top=0.97)
    ax = fig.add_subplot(111)
    ax.axis('off')
    if stats_type == 'median':
        if topic == 'frequency':
            cmap = cm.RdBu_r
        else:
            cmap = cm.RdBu
        im = bm.imshow(src[::-1], vmin=-_max, vmax=_max, cmap=cmap)
    elif stats_type == 'std' or stats_type == 'iqr':
        im = bm.imshow(src[::-1], vmin=0, vmax=_max, cmap=cm.RdPu)
    bm.drawcoastlines(linewidth=0.2)
    cax = fig.add_axes((0.1, 0.22, 0.8, 0.03))
    cb = plt.colorbar(im, cax=cax, orientation='horizontal', pad=0.05, aspect=45)
    cb.set_label(f'[{unit}]')
    cb.outline.set_linewidth(0)
    # save
    figname = f'map_{stats_type}.{scenario}.{season}.{variable}.{topic}.'
    figure_directory = os.path.join(figure_directory_main, diff_type, season)
    if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
    for suffix in suffixes:
        figpath = os.path.join(figure_directory, figname+suffix)
        plt.savefig(figpath)
        print(f'savefig: {figpath}')
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def draw_scatter(src1, src2, topic1, topic2, variable, stats_type, scenario, season):

    print(f'--- @draw_scatter ({variable}: {topic1} and {topic2}))')

    src1 = src1.flatten()
    src2 = src2.flatten()
    unit1 = find_unit(topic1, variable, season)
    unit2 = find_unit(topic2, variable, season)
    # remove outliers
    src1 = np.ma.masked_greater(src1, 1e+3)
    src2 = np.ma.masked_greater(src2, 1e+3)

    if topic1 != 'frequency':
        _min2, _max2 = src2.min(), src2.max()
        xs = np.array([_min2, _max2])
        slope = 1
        ys = slope * xs + 0

    fig = plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0.2, right=0.99, bottom=0.13, top=0.95)
    ax = fig.add_subplot(111)
    ax.axvline(x=0, color='#a0a0a0', zorder=2)
    ax.axhline(y=0, color='#a0a0a0', zorder=2)
    ax.scatter(src2, src1, s=1, marker='o', alpha=0.2, zorder=2.5)

    # xlim, ylim
    if diff_type == 'clim_diff':
        if stats_type == 'median':
            if topic1 == 'frequency':
                if season == 'ANN':
                    ax.set_ylim(top=30*12)
                else:
                    ax.set_ylim(top=30*3)
            else:  # apply consistent limits
                _min, _max = np.min((src1.min(), src2.min())), np.max((src1.max(), src2.max()))
                ax.plot(xs, ys, linestyle='--', linewidth=1, color='#a0a0a0', markersize=0, zorder=3)
                ax.set_xlim([_min, _max])
                ax.set_ylim([_min, _max])
        elif stats_type == 'std' or stats_type == 'iqr':
            if topic1 == 'frequency':
                ax.set_ylim(bottom=np.percentile(src1.compressed(), 0.01), top=np.percentile(src1.compressed(), 99.99))
                ax.set_xlim(left=0, right=np.percentile(src2.compressed(), 99.99))
            else:
                ax.plot(xs, ys, linestyle='--', linewidth=1, color='#a0a0a0', markersize=0, zorder=3)
                ax.set_ylim(bottom=0, top=np.max([np.percentile(src1.compressed(), 99.99), np.percentile(src2.compressed(), 99.99)]))
                ax.set_xlim(left=0, right=np.max([np.percentile(src1.compressed(), 99.99), np.percentile(src2.compressed(), 99.99)]))
    elif diff_type == 'clim_diff_pct':
        if stats_type == 'median':
            if topic1 == 'frequency':
                ax.set_ylim(bottom=np.percentile(src1.compressed(), 0.01), top=np.percentile(src1.compressed(), 99.99))
                ax.set_xlim(right=np.percentile(src2.compressed(), 99.99))
            else:
                ax.set_ylim(top=np.max([np.percentile(src1.compressed(), 99.99), np.percentile(src2.compressed(), 99.99)]))
                ax.set_xlim(right=np.max([np.percentile(src1.compressed(), 99.99), np.percentile(src2.compressed(), 99.99)]))
        elif stats_type == 'std' or stats_type == 'iqr':
            if topic1 == 'frequency':
                ax.set_ylim(bottom=0, top=np.percentile(src1.compressed(), 99.99))
                ax.set_xlim(left=0, right=np.percentile(src2.compressed(), 99.99))
            else:
                ax.set_ylim(bottom=0, top=np.max([np.percentile(src1.compressed(), 99.99), np.percentile(src2.compressed(), 99.99)]))
                ax.set_xlim(left=0, right=np.max([np.percentile(src1.compressed(), 99.99), np.percentile(src2.compressed(), 99.99)]))

    ax.set_xlabel(f'{topic2}  [{unit2}]')
    ax.set_ylabel(f'{topic1}  [{unit1}]')
    ax.axvline(x=0, color='#a0a0a0')

    figname = f'scatter_{stats_type}.{scenario}.{season}.{variable}.{topic2}-{topic1}.'
    figure_directory = os.path.join(figure_directory_main, diff_type, season)
    if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
    for suffix in suffixes:
        figpath = os.path.join(figure_directory, figname+suffix)
        plt.savefig(figpath, dpi=300)
        print(f'savefig: {figpath}')
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def find_inconsistent(src1, src2): 

    src = np.full((ny,nx), 1e+20)
    # 1.5-2.5
    #  |   |
    # 0.5-3.5
    src[np.where((src1 ==0)  & (src2 ==0)) ] = -0.5
    src[np.where((src1 < 0)  & (src2 < 0)) ] = 0.5
    src[np.where((src1 < 0)  & (src2 > 0)) ] = 1.5
    src[np.where((src1 > 0)  & (src2 > 0)) ] = 2.5
    src[np.where((src1 > 0)  & (src2 < 0)) ] = 3.5
    src = np.ma.masked_equal(src, 1e+20)
    src = np.ma.masked_equal(src, -0.5)
    return src


# ----------------------------------------------------------------------------------------------------------------------
def draw_inconsistence_map(src, variable1, variable2, comparison, ensemble_type, topic, scenario, season):

    print(f'--- @draw_inconsistence_map ({topic}: {variable1} and {variable2}))')

    src = np.ma.masked_array(src, mask=landseamask)[11:293]

    fig1 = plt.figure(num=1, figsize=(4, 1.7))
    gs = gridspec.GridSpec(1, 1)  # (rows,cols)
    gs.update(left=0.01, right=0.99, bottom=0.02, top=0.98, hspace=0.02, wspace=0.01)
    # ax1 (Upper left: Main ensemble value)
    ax1 = plt.subplot(gs[0,0])
    plt.sca(ax1)
    ax1.axis('off')
    ax_pos = ax1.get_position()
    norm1 = mpl.colors.Normalize()
    # draw a map
    bounds = [0., 1., 2., 3., 4.]
    if topic == 'frequency':
        colors = np.divide([[69., 64., 145.], [255., 255.,  51.], [241., 86., 32.], [ 32., 198., 192.]], 255.)  # purple, yellow, orange, bluegreen
    else:
        colors = np.divide([[241., 86., 32.], [32., 198., 192.], [69., 64., 145.], [255., 255., 51.]], 255.)  # orange, bluegreen, purple, yellow
    cmap = mpl.colors.ListedColormap(colors)
    bm.imshow(src[::-1], norm=norm1, cmap=cmap, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
    bm.drawcoastlines(linewidth=0.2)
    # colorbar
    box_unit, dx, dy, pad = 0.07, 0.1, 0.12,  0.01 
    _bounds = [0., 1., 2.]
    _colors1 = [colors[0], colors[3]]
    _colors2 = [colors[1], colors[2]]
    cmap1 = mpl.colors.ListedColormap(_colors1)
    cmap2 = mpl.colors.ListedColormap(_colors2)
    ax11 = fig1.add_axes([ax_pos.x0+dx, ax_pos.y0+dy,          box_unit, box_unit])
    ax12 = fig1.add_axes([ax_pos.x0+dx, ax_pos.y0+dy+box_unit, box_unit, box_unit])
    # upper
    norm = mpl.colors.BoundaryNorm(_bounds, cmap2.N)
    cb2 = mpl.colorbar.ColorbarBase(ax12, cmap=cmap2, norm=norm, boundaries=_bounds, spacing='uniform', orientation='horizontal')
    cb2.set_ticks(_bounds)
    cb2.set_ticklabels([])
    cb2.ax.tick_params(labelsize=3,direction='in')
    # lower
    norm = mpl.colors.BoundaryNorm(_bounds, cmap1.N)
    cb1 = mpl.colorbar.ColorbarBase(ax11, cmap=cmap1, norm=norm, boundaries=_bounds, spacing='uniform', orientation='horizontal')
    cb1.set_ticks(_bounds)
    cb1.set_ticklabels([])
    cb1.ax.tick_params(labelsize=3,direction='in')
    # label
    fontsize = 10
    if topic == 'frequency':
        label1 = dict_standardizedname[variable1]
        label2 = dict_standardizedname[variable2]
    else:
        label1 = dict_varname[variable1]
        label2 = dict_varname[variable2]
    fig1.text(ax_pos.x0+dx-pad,          ax_pos.y0+dy+box_unit,       f'{label1} -', va='center', ha='right',  fontsize=fontsize)
    fig1.text(ax_pos.x0+dx+box_unit+pad, ax_pos.y0+dy+box_unit,       f'{label1} +', va='center', ha='left',   fontsize=fontsize)
    fig1.text(ax_pos.x0+dx+box_unit/2,   ax_pos.y0+dy-pad,            f'{label2} -', va='top',    ha='center', fontsize=fontsize)
    fig1.text(ax_pos.x0+dx+box_unit/2,   ax_pos.y0+dy+2*box_unit+pad, f'{label2} +', va='bottom', ha='center', fontsize=fontsize)
    # save
    figname = f'map.inconsistentSign-{comparison}.{ensemble_type}.{scenario}_{season}.{topic}.{variable1}-{variable2}.'
    figure_directory = os.path.join(figure_directory_main, diff_type, season)
    if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
    for suffix in suffixes:
        figpath = os.path.join(figure_directory, figname+suffix)
        plt.savefig(figpath)
        print(f'savefig: {figpath}')
    plt.close(1)


# ----------------------------------------------------------------------------------------------------------------------
def main():

    combinations = itertools.combinations(variables, 2)
    for scenario, stats_type, season in itertools.product(scenarios, stats_types, seasons):
        print(f'\n\n======= a process for {scenario} {stats_type} {season} =======\n')
       
        dict_maps = {variable: {} for variable in variables}
        for variable in variables:

            average       = gen_src('average',       variable, scenario, stats_type, season)  # (ny, nx)
            percentile_10 = gen_src('percentile_10', variable, scenario, stats_type, season)  # (ny, nx)
            frequency     = gen_src('frequency',     variable, scenario, stats_type, season)  # (ny, nx)
            dict_maps[variable]['average']       = average
            dict_maps[variable]['percentile_10'] = percentile_10
            dict_maps[variable]['frequency']     = frequency

            # draw maps
            if MAP:
                for topic in topics:
                    draw_map(dict_maps[variable][topic], topic, variable, stats_type, scenario, season)
            # draw scatter
            if SCATTER:
                for topic1, topic2 in [('average', 'percentile_10'), ('frequency', 'percentile_10')]:
                    draw_scatter(dict_maps[variable][topic1], dict_maps[variable][topic2], 
                                 topic1, topic2, 
                                 variable, stats_type, scenario, season)

        if MAP and stats_type == 'median' :
            for (variable1, variable2), topic in itertools.product(combinations, topics):
                print(f'\n find_inconsistent: {topic}, {variable1}-{variable2}')
                src1 = dict_maps[variable1][topic]
                src2 = dict_maps[variable2][topic]
                src = find_inconsistent(src1, src2)
                draw_inconsistence_map(src, variable1, variable2, 'variable', stats_type, topic, scenario, season)           

    print('Congratulations!! Successfully this process finished!!   d(^o^)b')


if __name__ == '__main__':
    main()
