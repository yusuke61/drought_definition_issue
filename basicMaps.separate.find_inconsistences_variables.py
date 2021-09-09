# To:  detect regions where sign of change is inconsistent between drought categories... >>> Fig.3, SupFig.8-10
#           - between variables  (eg. pr vs soilmoist)
#           - between druoght metrix (n_month vs n_event)
#       - Global maps: base-period, historical, 2050s, and 2080s
#           - absolute or change  x  ensemble or members
#               - a change map has 2D colorbar (change & agreement/
# By: Yusuke Satoh (NIES)

import os
import sys
import itertools
import numpy as np
import datetime
from numpy import array, ma, divide, mean, median, fromfile, where, full
import matplotlib.pyplot as plt
import matplotlib as mpl
from netCDF4 import Dataset
from matplotlib import colors, gridspec
from mpl_toolkits.basemap import Basemap
from utiltools import fillinMasked, extractMasked, flipud
from os.path import join as genpath

today = datetime.date.today().strftime('%Y%m%d')
hostname = os.uname()[1]


# select for your interest ----------------------------------------------------------------------------------------------------------------------
TEST = False  # default
#TEST = True

comparison = 'variable'
#comparison = 'proxy'

variables = ['pr', 'qtot', 'soilmoist']

#proxies = ['period_total_drought_months', 'period_total_number_of_events']
proxies = ['period_total_drought_months']

ghms = ['matsiro', 'cwatm', 'clm45', 'h08', 'jules-w1', 'lpjml', 'watergap2']

gcms  = ['hadgem2-es','ipsl-cm5a-lr','gfdl-esm2m','miroc5']

scenarios  = ['rcp26', 'rcp60', 'rcp85']
#scenarios  = ['rcp85']

#scales = [1, 3, 6, 12]
scales = [3, 12]
#scales = [1]
#scales = [3]

if TEST:
    ghms = ['matsiro', 'cwatm']
    gcms  = ['hadgem2-es']
    scenarios = ['rcp85']
    scales = [3]

#ensemble_types = ['median','mean']
ensemble_types = ['median']
#ensemble_types = ['mean']

#distributions = ['gamma', 'pearson']
distributions = ['gamma']

drought_severities = ['severe']
#drought_severities = ['mild', 'moderate', 'severe', 'extreme']

climatology = '30yrs'; periods = ['base_period', 'recent30yrs', 'nearfuture', 'farfuture']
#climatology = '50yrs'; periods = ['base_period', '1st-half-21C', '2nd-half-21C']

DUMMY = False
EnsmblMap = True
ghmsMap = True
membermap = False
s2nmap = False

projection = 'cyl'
#projection = 'robin'
#projection = 'eck4'


# Basically, you don't need to edit hear -------------------------------------------------------------------------------
base = 'base_1861-1960'
version = 'v2'

syear = 1861
eyear = 2099
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
    dict_period = {
        'base_period': (1861, 1890),  # 30yrs
        'recent30yrs': (1990, 2019),  # 30yrs
        'nearfuture':  (2035, 2064),  # 30yrs
        'farfuture':   (2070, 2099),  # 30yrs
        }
elif climatology == '50yrs':
    dict_period = {
        'base_period':  (1861, 1890),  # 50yrs
        '1st-half-21C': (2000, 2049),  # 50yrs
        '2nd-half-21C': (2050, 2099),  # 50yrs
        }
years_full_period = range(1861, 2099+1)
period_indices = [(years_full_period.index(dict_period[period][0]), years_full_period.index(dict_period[period][1])) for period in periods]

dict_unit = {
    'period_total_drought_months':   ('months', 1, 'months'),
    'period_total_number_of_events': ('times', 1, 'times'),
    }

ks = 0.05    # 95% level significance

savefig_dpi = 300
suffixes = ['png', 'pdf']

if 'scs' in hostname: data_directory_top = '/data/rg001/sgec0017/data' 

data_directory_main = os.path.join(data_directory_top, 'isimip2b.standardized_drought',
                                   'climate_indices_postprocessed',
                                   base, version,
                                   f'climatology_{climatology}')
figure_directory_main = os.path.join(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                     'basicMaps.separate.find_inconsisnteces_variables',
                                     #base, version,
                                     #'climatology_{}'.format(climatology),
                                     f'clm{climatology}',
                                     #today
                                     )

lndmskPath = os.path.join(data_directory_top, 'mapmask', 'ISIMIP2b_landseamask_generic.nc4')
grlmskPath = os.path.join(data_directory_top, 'mapmask', 'GAUL/flt/gaul2014_05deg.flt')      # GreenLand is 98
grdaraPath = os.path.join(data_directory_top, 'mapmask', 'grd_ara.hlf')

nghm, ngcm = len(ghms), len(gcms)
nmembers = nghm*ngcm
ny, nx, ntime = 360, 720, 2868

grl_mask = ma.masked_equal(fromfile(grlmskPath, 'float32').reshape(360,720),98).mask
lnd_mask = Dataset(lndmskPath)['LSM'][:][0].mask
lnd_mask = ma.mask_or(lnd_mask, grl_mask)
nlndgrid = 360*720 - lnd_mask.sum()
area = ma.masked_array(fromfile(grdaraPath, 'float32').byteswap().reshape(360,720), mask=lnd_mask)

#resolution = 'i'
resolution = 'l'
if projection == 'cyl':
    bm = Basemap(projection=projection,llcrnrlat=-56.5,urcrnrlat=84.5,llcrnrlon=-180.,urcrnrlon=180.,resolution=resolution) # correct
elif projection == 'robin' or projection == 'eck4':
    bm = Basemap(projection=projection, lon_0=0, resolution=resolution)

dict_standerdizedname = {'pr': 'SPI', 'qtot': 'SRI', 'soilmoist': 'SSI'}


# ----------------------------------------------------------------------------------------------------------------------
def read_netcdf(item_compared_about, item_investigate_about, drought_severity, scale, distribution, ghm, gcm, scenario):

    if not DUMMY:  # src.shape (ny, nx, ntime)  ntime = 2868 = 12*239

        if   comparison == 'variable':
            variable, proxy = item_compared_about, item_investigate_about
        elif comparison == 'proxy':
           variable, proxy = item_investigate_about, item_compared_about

        if variable == 'pr':
            filename = f'{gcm}_hist{scenario}_pr_monthly_1861_2099_spi_{distribution}_{scale:02}_{drought_severity}_{proxy}.nc'
        else:
            soc_hist, soc_future = dict_soc[ghm]
            filename = f'{ghm}_{gcm}_hist{scenario}_{soc_hist}_{soc_future}_co2_{variable}_monthly_1861_2099_spi_{distribution}_{scale:02}_{drought_severity}_{proxy}.nc'
        srcpath = genpath(data_directory_main, variable, filename)

        if not os.path.isfile(srcpath):
            #print(f'????: {srcpath} is NOT exist... Check!! Caution!!!!!!')
            #srcs  = full((len(periods), 360, 720), 1.e+20)
            raise FileNotFoundError(srcpath)
        else:
            srcs = Dataset(srcpath)[proxy][:]  # (nperiod, ny, nx)
            print(f'read: {srcpath} {src.shape}')
            srcs[np.isnan(srcs)] = 1e+20
            srcs = np.ma.masked_equal(srcs, 1e+20)
        return srcs
    else:
        print(f'generate DUMMY src...  ({len(periods)}, 30, {nlndgrid})')
        return np.random.rand(len(periods), 360, 720)


# ----------------------------------------------------------------------------------------------------------------------
def write2nc(src, item1, item2, comparison, item_investigate_about,
             scenario, period, ensemble_type, distribution, scale, drought_severity):

    baseyear = 1661
    complevel = 5
    lats = np.arange(89.75, -90, -0.5)
    lons = np.arange(-179.75, 180, 0.5)
    Times = [(datetime.date(dict_period[period][0],1,1)-datetime.date(baseyear,1,1)).days]
    nT = len(Times)
    print(f'lats:  {len(lats)}\nlons:  {len(lons)}\nnT:   {len(Times)}')

    if not src.shape[-2] == ny:
        _src = np.full((360,720), 1e+20)  # missing_value... 
        _src[11:293,:] = src.filled(1e+20)
        src = np.ma.masked_equal(_src, 1e+20)
        del _src

    src = src.reshape(-1, ny, nx)

    # open a netcdf and write-in
    filename = f'basicMap.ensemble{ensemble_type}_inconsisntentSign-{comparison}_{item1}-{item2}__{item_investigate_about}_{scenario}_{period}__{distribution}_{drought_severity}.nc'
    output_directory = genpath(figure_directory_main, scenario, ensemble_type, drought_severity, distribution, str(scale))
    if not os.path.isdir(output_directory): os.makedirs(output_directory)
    outputpath = genpath(output_directory, filename)

    rootgrp = Dataset(outputpath, 'w', format='NETCDF4')
    rootgrp.description = 'ISIMIP2b drought propagetion analysis'
    import time
    rootgrp.history     = 'Created ' + time.ctime(time.time())
    rootgrp.source      = 'ISIMIP2b'
    rootgrp.title       = f'agree-disagree map: {item1}&{item2} ({period})'
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

    srcs                     = rootgrp.createVariable('agree-disagree',
                                                      'f4', ('time','lat','lon'),
                                                      zlib=True, complevel=complevel,
                                                      fill_value=1.e+20,
                                                      chunksizes=(1, ny, nx)
                                                      )
    srcs.missing_value       = np.float32(1.e+20)
    srcs.memo                = '1.5-2.5\n |   |\n0.5-3.5'
    srcs.memo1               = 'drought type: ' + ', '.join([item1, item2])
    srcs.memo2               = 'scenario    : ' + ', '.join(scenario)
    srcs.memo3               = 'gcm         : ' + ', '.join(gcms)
    srcs.memo4               = 'ghm         : ' + ', '.join(ghms)

    times[:]      = Times
    latitudes[:]  = lats
    longitudes[:] = lons
    srcs[:]       = src

    rootgrp.close()
    print(f'\nFinished writting   : {outputpath} {src.shape} {src.min()}-{src.max()}\n')


# ----------------------------------------------------------------------------------------------------------------------
def draw_inconsistence_map(src, significance_mask, item1, item2, period,
                           comparison, ensemble_type, item_investigate_about, 
                           scenario, distribution, scale, drought_severity):

    fig1 = plt.figure(num=1, figsize=(4, 1.7))
    gs = gridspec.GridSpec(1, 1)  # (rows,cols)
    gs.update(left=0.01, right=0.99, bottom=0.02, top=0.98, hspace=0.02, wspace=0.01)
    # ax1 (Upper left: Main ensemble value) -------------------------
    ax1 = plt.subplot(gs[0,0])
    ax1.axis('off')
    ax_pos = ax1.get_position()
    norm1 = mpl.colors.Normalize()
    # draw a map
    bounds = [-1., 0., 1., 2., 3., 4.]
    # both insignificant changes
    _src = flipud(np.ma.masked_array(src, mask=~significance_mask))  # mask-out area with significant changes
    colors1 = divide([[188, 154, 144], [152., 148., 201.], [ 255., 255., 182.], [ 241., 163., 135.], [142., 219., 216.]], 255.)  # purple, yellow, orange, bluegreen
    cmap1 = mpl.colors.ListedColormap(colors1)
    ims = bm.imshow(_src, norm=norm1, cmap=cmap1, vmin=bounds[0], vmax=bounds[-1], interpolation='nearest')
    # both or either of significant changes
    _src = flipud(np.ma.masked_array(src, mask=significance_mask))  # mask-out area with insignificant changes
    colors2 = divide([[188, 154, 144], [22., 15., 121.], [ 230., 230., 0.], [ 199., 56., 4.], [0., 148., 143.]], 255.)  # purple, yellow, orange, bluegreen
    cmap2 = mpl.colors.ListedColormap(colors2)
    ims = bm.imshow(_src, norm=norm1, cmap=cmap2, vmin=bounds[0], vmax=bounds[-1], interpolation='nearest')
    bm.drawcoastlines(linewidth=0.2)
    # colorbar
    if scenario == 'rcp85' and scale == 3 and period == 'farfuture':
        box_unit, dx, dy, pad = 0.07, 0.1, 0.15,  0.01
    else:
        box_unit, dx, dy, pad = 0.095, 0.1, 0.15,  0.01 
    _bounds = [0., 0.5, 1., 1.5, 2.]
    length = 0.
    width = 0.
    linewidth = 0.
    _colors1 = [colors2[1], colors2[1], colors2[4], colors2[4]]
    _colors2 = [colors2[1], colors1[1], colors1[4], colors2[4]]
    _colors3 = [colors2[2], colors1[2], colors1[3], colors2[3]]
    _colors4 = [colors2[2], colors2[2], colors2[3], colors2[3]]
    cmap1 = mpl.colors.ListedColormap(_colors1)
    cmap2 = mpl.colors.ListedColormap(_colors2)
    cmap3 = mpl.colors.ListedColormap(_colors3)
    cmap4 = mpl.colors.ListedColormap(_colors4)
    ax14 = fig1.add_axes([ax_pos.x0+dx, ax_pos.y0+dy+box_unit*1.5, box_unit*0.9, box_unit/2])
    ax13 = fig1.add_axes([ax_pos.x0+dx, ax_pos.y0+dy+box_unit*1.0, box_unit*0.9, box_unit/2])
    ax12 = fig1.add_axes([ax_pos.x0+dx, ax_pos.y0+dy+box_unit*0.5, box_unit*0.9, box_unit/2])
    ax11 = fig1.add_axes([ax_pos.x0+dx, ax_pos.y0+dy+box_unit*0.0, box_unit*0.9, box_unit/2])
    # most upper
    norm = mpl.colors.BoundaryNorm(_bounds, cmap4.N)
    cb4 = mpl.colorbar.ColorbarBase(ax14, cmap=cmap4, norm=norm, boundaries=_bounds, spacing='uniform', orientation='horizontal')
    cb4.set_ticks(_bounds)
    cb4.set_ticklabels([])
    cb4.ax.tick_params(labelsize=3, direction='in', length=length, width=width)
    cb4.outline.set_linewidth(linewidth)
    # middle upper
    norm = mpl.colors.BoundaryNorm(_bounds, cmap3.N)
    cb3 = mpl.colorbar.ColorbarBase(ax13, cmap=cmap3, norm=norm, boundaries=_bounds, spacing='uniform', orientation='horizontal')
    cb3.set_ticks(_bounds)
    cb3.set_ticklabels([])
    cb3.ax.tick_params(labelsize=3, direction='in', length=length,width=width)
    cb3.outline.set_linewidth(linewidth)
    # middle lower
    norm = mpl.colors.BoundaryNorm(_bounds, cmap2.N)
    cb2 = mpl.colorbar.ColorbarBase(ax12, cmap=cmap2, norm=norm, boundaries=_bounds, spacing='uniform', orientation='horizontal')
    cb2.set_ticks(_bounds)
    cb2.set_ticklabels([])
    cb2.ax.tick_params(labelsize=3, direction='in', length=length, width=width)
    cb2.outline.set_linewidth(linewidth)
    # most lower
    norm = mpl.colors.BoundaryNorm(_bounds, cmap1.N)
    cb1 = mpl.colorbar.ColorbarBase(ax11, cmap=cmap1, norm=norm, boundaries=_bounds, spacing='uniform', orientation='horizontal')
    cb1.set_ticks(_bounds)
    cb1.set_ticklabels([])
    cb1.ax.tick_params(labelsize=3, direction='in', length=length,width=width)
    cb1.outline.set_linewidth(linewidth)
    # label
    if scenario == 'rcp85' and scale == 3 and period == 'farfuture': fontsize = 7
    else:                                                            fontsize = 9.5
    stdname1, stdname2 = f'{dict_standerdizedname[item1]}{scale}', f'{dict_standerdizedname[item2]}{scale}'
    fig1.text(ax_pos.x0+dx-pad,              ax_pos.y0+dy+box_unit,       f'{stdname1}--',   va='center', ha='right',  fontsize=fontsize)
    fig1.text(ax_pos.x0+dx+box_unit*0.9+pad, ax_pos.y0+dy+box_unit,       f'+{stdname1}',    va='center', ha='left',   fontsize=fontsize)
    fig1.text(ax_pos.x0+dx+box_unit/2,       ax_pos.y0+dy-pad/2,          f'--\n{stdname2}', va='top',    ha='center', fontsize=fontsize, linespacing=1.0)
    fig1.text(ax_pos.x0+dx+box_unit/2,       ax_pos.y0+dy+2*box_unit+pad, f'{stdname2}\n+',  va='bottom', ha='center', fontsize=fontsize, linespacing=0.7)
    # save
    figurename = f'basicMap.ensemble{ensemble_type}_inconsisntentSign-{comparison}_{item1}-{item2}__{item_investigate_about}_{scenario}_{period}__{distribution}_{scale}_{drought_severity}.'
    figure_directory = genpath(figure_directory_main, scenario, ensemble_type, drought_severity, distribution, str(scale))
    if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
    figurepath = genpath(figure_directory, figurename)
    for suffix in suffixes:
        if os.path.exists(figurepath+suffix):
            print('File exists, will be overwritten.')
        plt.savefig(figurepath+suffix, dpi=savefig_dpi)
    print('savefig: {}\n'.format(figurepath+suffix))
    plt.close(1)


# ----------------------------------------------------------------------------------------------------------------------
def main(*args):

    strTIME = datetime.datetime.now()
    print('START basicMap.separate.find_inconsistences.py\n@{}\n'.format(strTIME.strftime("%Y-%m-%d %H:%M:%S")))

    if comparison == 'variable':
        items_compared_about, items_investigate_about = variables, proxies
    elif comparison == 'proxy':
        items_compared_about, items_investigate_about = proxies, variables
    combinations = list(itertools.combinations(items_compared_about, 2))

    for item_investigate_about, scenario, ensemble_type, distribution, scale, drought_severity in itertools.product(items_investigate_about, 
                                                                                                                    scenarios, ensemble_types, 
                                                                                                                    distributions, scales, drought_severities):
        print('\nJob started !!!')
        print('\n\n===================\n {} {} {} {} {} {}\n===================\n'.format(scenario, ensemble_type,
                                                                                          distribution, scale,
                                                                                          drought_severity,
                                                                                          item_investigate_about))
        strTime = datetime.datetime.now()
        print(strTIME.strftime("%Y-%m-%d %H:%M:%S"))
        dict_ensemble_data = {}
        dict_significance_indevidual = {}
        for item_compared_about in items_compared_about:
                        
            # Reading data
            srcs = array([[read_netcdf(item_compared_about, item_investigate_about, drought_severity, scale, distribution, ghm, gcm, scenario) for ghm in ghms] for gcm in gcms])  # (ngcm, nghm, nperiod, ny, nx)
            srcs = ma.masked_equal(srcs, 1e+20)         # orignally, missing_value in each data is 1e+20
            srcs = np.transpose(srcs, axes=(2,0,1,3,4))
            print(f'srcs.shape: {srcs.shape}')  # (nperiod, ngcm, nghm, ny, nx)

            # make climatology for each combination of GHM&GCM  (for each ensemble members)
            climatology_base_period = srcs[0]  #      (ngcm,nghm,ny,nx)
            climatology_periods = srcs[1:]     # (nperiod,ngcm,nghm,ny,nx)
            del srcs

            # percentage change (change_pc) for each ensemble members [%]
            climatological_differences_percent = np.divide(climatology_periods-climatology_base_period, climatology_base_period) * 100   # (nperiod,ngcm,nghm,ny,nx) [%]

            # mmed_climatological_differences_percent : change >> ensembling
            if   ensemble_type == 'median':
                mmed_climatological_differences_percent = median(climatological_differences_percent, axis=(1,2))  # (nperiod,ny,nx) [%]
            elif ensemble_type == 'mean':
                mmed_climatological_differences_percent = mean(climatological_differences_percent, axis=(1,2))    # (nperiod,ny,nx) [%]
            else:
                raise ValueError('Warning. Check ensemble_type.')
            # an item data to compare got ready here
            dict_ensemble_data[item_compared_about] = ma.masked_equal(mmed_climatological_differences_percent, 1e+20)  # (nperiod,ny,nx)
            dict_significance_indevidual[item_compared_about] = ma.make_mask(np.abs(dict_ensemble_data[item_compared_about]) < 10)  # masked, if < 10%

        dict_consistences = {}
        dict_significance_combined = {}
        for (item1, item2) in combinations:
            print(item1, item2)
            src = full(dict_ensemble_data[item1].shape, 1e+20)
            # 1.5-2.5
            #  |   |
            # 0.5-3.5
            src[where((dict_ensemble_data[item1].filled(0) ==0)  | (dict_ensemble_data[item2].filled(0) ==0))] = -0.5
            src[where((dict_ensemble_data[item1].filled(0) < 0)  & (dict_ensemble_data[item2].filled(0) < 0))] = 0.5
            src[where((dict_ensemble_data[item1].filled(0) < 0)  & (dict_ensemble_data[item2].filled(0) > 0))] = 1.5
            src[where((dict_ensemble_data[item1].filled(0) > 0)  & (dict_ensemble_data[item2].filled(0) > 0))] = 2.5
            src[where((dict_ensemble_data[item1].filled(0) > 0)  & (dict_ensemble_data[item2].filled(0) < 0))] = 3.5
            src = ma.masked_equal(src, 1e+20)
            #src = ma.masked_equal(src, -0.5)
            src = ma.masked_array(src, mask=np.resize(lnd_mask, src.shape))
            dict_consistences[f'{item1}-{item2}'] = src  # (nperiod,nY,nX)
            dict_significance_combined[f'{item1}-{item2}'] = np.ma.mask_or(dict_significance_indevidual[item1], dict_significance_indevidual[item2])  # or

        # Make figure
        print('\nFigure making...')
        for (item1, item2), (i_period, period) in itertools.product(combinations, enumerate(periods[1:])):
            print(f' >>>> {item1} vs {item2}: {period} ({i_period})')

            src = dict_consistences[f'{item1}-{item2}'][i_period]  # 0-4
            significance_mask = dict_significance_combined[f'{item1}-{item2}'][i_period]  # masked if change rate < 10%
            if projection == 'cyl': 
                src = src[11:293,:]
                significance_mask = significance_mask[11:293,:]
            draw_inconsistence_map(src, significance_mask,
                                   item1, item2, period,  
                                   comparison, ensemble_type, item_investigate_about, 
                                   scenario, distribution, scale, drought_severity)
            write2nc(src, item1, item2, comparison, item_investigate_about,
                     scenario, period, ensemble_type, distribution, scale, drought_severity)

        endTime = datetime.datetime.now()
        diffTime = endTime - strTime
        print('end @', endTime.strftime("%Y-%m-%d %H:%M:%S"))
        print('took {} min in total.'.format(int(diffTime.seconds/60)))

    print('This process was successfully DONE!!   d(^o^)b')


if __name__=='__main__':
    main(*sys.argv)


