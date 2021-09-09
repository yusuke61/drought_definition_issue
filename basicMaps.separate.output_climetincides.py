# To:   This is for basic maps (pr, qtot, soilmoist).   >>> Fig.1, SupFig.1
#       - Global maps: base-period, historical, 2050s, and 2080s
#           - absolute or change  x  ensemble or members
#               - a change map has 2D colorbar (change & agreement/
# By: Yusuke Satoh (NIES)

import os
import sys
import itertools
import numpy as np
import pandas as pd
import datetime
from numpy import array, ma, divide, mean, median, fromfile, subtract, percentile, var, where, full
import matplotlib.pyplot as plt
import matplotlib as mpl
from netCDF4 import Dataset
from matplotlib import cm, colors, gridspec
from mpl_toolkits.basemap import Basemap
from utiltools import fillinMasked, extractMasked, flipud
from scipy import stats
from os.path import join as genpath

hostname = os.uname()[1]
today = datetime.date.today().strftime('%Y%m%d')

# select for your interest ---------------------------------------------------------------------------------------------
TEST = False

KStest = True; ks_test = 'ON'
#KStest = False; ks_test = 'OFF'

variables = [sys.argv[1]]
#variables = ['pr', 'qtot', 'soilmoist']
#variables = ['pr']
#variables = ['soilmoist']
#variables = ['qtot']

scenarios = [sys.argv[2]]
#scenarios  = ['rcp26', 'rcp60', 'rcp85']
#scenarios  = ['rcp85']

_ghms = ['matsiro', 'cwatm', 'clm45', 'h08', 'jules-w1', 'lpjml', 'watergap2']
gcms  = ['hadgem2-es', 'ipsl-cm5a-lr', 'gfdl-esm2m', 'miroc5']

if TEST: 
    variables = ['soilmoist']
    scenarios = ['rcp26']
    gcms  = ['hadgem2-es', 'ipsl-cm5a-lr']
    _ghms = ['matsiro', 'cwatm']

#ensemble_types = ['median','mean']
ensemble_types = ['median']
#ensemble_types = ['mean']

#distributions = ['gamma', 'pearson']
distributions = ['gamma']

#scales = [1, 3, 6, 12]
#scales = [1, 6, 12]
#scales = [1]
scales = [3]
#scales = [12]

drought_severities = ['severe']
#drought_severities = ['mild', 'moderate', 'severe', 'extreme']

#names = ['period_total_drought_months', 'period_total_number_of_events']
names = ['period_total_drought_months']

# Caution!! Need to be consistent with data!!! 09.climateindices_postprocess.go.sh
# You can specify output period at the Loop.
climatology = '30yrs'; periods = ['base_period', 'recent30yrs', 'nearfuture', 'farfuture']  
#climatology = '50yrs'; periods = ['base_period', '1st-half-21C', '2nd-half-21C']

DUMMY = False
EnsmblMap = True
ghmsMap = False
membermap = False
s2nmap = False
UncertaintySource = False

projection = 'cyl'
#projection = 'robin'
#projection = 'eck4'
savefig_dpi = 300
suffixes = ['png', 'pdf']


# Basically, you don't need to edit hear -----------------------------------------------------------------
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
agreeThrsh = [0.8,0.6]
#agreeThrsh = [0.8,0.7,0.6]

dict_standerdizedname = {'pr': 'SPI', 'qtot': 'SRI', 'soilmoist': 'SSI'}

if 'scs' in hostname: data_directory_top = '/data/rg001/sgec0017/data' 
data_directory_main = os.path.join(data_directory_top, 'isimip2b.standardized_drought',
                                   'climate_indices_postprocessed', base, version, f'climatology_{climatology}')
figure_directory_main = os.path.join(data_directory_top, 'figure_box', 'isimip2b.standardized_drought',
                                     #'basicMaps.separate.output_climetincides', base, version, f'climatology_{climatology}_KS-{ks_test}', today)
                                     'basicMaps.separate.output_climetincides', f'clm{climatology}_KS-{ks_test}')
if TEST: figure_directory_main = figure_directory_main + 'test'

lndmskPath = os.path.join(data_directory_top, 'mapmask', 'ISIMIP2b_landseamask_generic.nc4')
grlmskPath = os.path.join(data_directory_top, 'mapmask', 'GAUL/flt/gaul2014_05deg.flt') # GreenLand is 98
grdaraPath = os.path.join(data_directory_top, 'mapmask', 'grd_ara.hlf')
grl_mask = ma.masked_equal(fromfile(grlmskPath, 'float32').reshape(360,720),98).mask
lnd_mask = Dataset(lndmskPath)['LSM'][:][0].mask
lnd_mask = ma.mask_or(lnd_mask, grl_mask)
area = ma.masked_array(fromfile(grdaraPath, 'float32').byteswap().reshape(360,720), mask=lnd_mask)
nlndgrid = 360*720 - lnd_mask.sum()
ny, nx, ntime = 360, 720, 2868

#resolution = 'i'
resolution = 'l'
if projection == 'cyl':
    bm = Basemap(projection=projection,llcrnrlat=-56.5,urcrnrlat=84.5,llcrnrlon=-180.,urcrnrlon=180.,resolution=resolution)
elif projection == 'robin' or projection == 'eck4':
    bm = Basemap(projection=projection, lon_0=0, resolution=resolution)


# ----------------------------------------------------------------------------------------------------------------------
def read_netcdf(variable, name, drought_severity, scale, distribution, ghm, gcm, scenario):

    if not DUMMY:  # src.shape (ny, nx, ntime)  ntime = 2868 = 12*239
        if variable == 'pr':
            filename = f'{gcm}_hist{scenario}_pr_monthly_1861_2099_spi_{distribution}_{scale:02}_{drought_severity}_{name}.nc'
        else:
            soc_hist, soc_future = dict_soc[ghm]
            filename = f'{ghm}_{gcm}_hist{scenario}_{soc_hist}_{soc_future}_co2_{variable}_monthly_1861_2099_spi_{distribution}_{scale:02}_{drought_severity}_{name}.nc'
        srcpath = genpath(data_directory_main, variable, filename)

        if not os.path.isfile(srcpath):
            #print(f'Caution!! {srcpath} is not exist... Check!!')
            #srcs = extractMasked(full((len(periods), 360, 720), 1.e+20, ), lnd_mask)
            raise FileNotFoundError(srcpath)
        else:
            srcs = Dataset(srcpath)[name][:]  # (nperiod, ny, nx)
            srcs[np.isnan(srcs)] = 1e+20
            srcs = extractMasked(srcs, lnd_mask)  # (nyear, nland)
            print(f'read: {srcpath} {srcs.shape}')
        return srcs  # (nperiod, nland)
    else:
        print(f'generate DUMMY src...  ({len(periods)}, 30, {nlndgrid})')
        return np.random.rand(len(periods), nlndgrid)


# ----------------------------------------------------------------------------------------------------------------------
def write2nc(src, what_this_is_about, scenario, period, ensemble_type, variable, distribution, scale, drought_severity):

    baseyear = 1661
    complevel = 5
    lats = np.arange(89.75, -90, -0.5)
    lons = np.arange(-179.75, 180, 0.5)
    Times = [(datetime.date(dict_period[period][0],1,1)-datetime.date(baseyear,1,1)).days]
    nT = len(Times)
    print(f'lats:  {len(lats)}\nlons:  {len(lons)}\nnT:   {len(Times)}')

    if not src.shape[-2] == ny:
        _src = np.full((360,720), 1e+20)
        _src[11:293,:] = src
        src = np.ma.masked_equal(_src, 1e+20)
        del _src

    src = src.reshape(-1, ny, nx)

    # open a netcdf and write-in
    filename = f'{what_this_is_about}_{variable}_{period}_{ensemble_type}_{distribution}_{scale}_{drought_severity}.nc'
    output_directory = genpath(figure_directory_main, scenario, ensemble_type, variable, distribution, str(scale))
    if not os.path.isdir(output_directory): os.makedirs(output_directory)
    outputpath = genpath(output_directory, filename)

    rootgrp = Dataset(outputpath, 'w', format='NETCDF4')
    rootgrp.description = 'ISIMIP2b drought propagetion analysis'
    import time
    rootgrp.history     = 'Created ' + time.ctime(time.time())
    rootgrp.source      = 'ISIMIP2b'
    rootgrp.title       = f'KS-test: {variable} {period}'
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

    srcs                     = rootgrp.createVariable(f'{what_this_is_about}_{variable}',
                                                      'f4', ('time','lat','lon'),
                                                      zlib=True, complevel=complevel,
                                                      fill_value=1.e+20,
                                                      chunksizes=(1, ny, nx)
                                                      )
    srcs.missing_value       = np.float32(1.e+20)
    srcs.memo1               = 'drought type: ' + ', '.join(variables)
    srcs.memo2               = 'scenario    : ' + ', '.join(scenarios)
    srcs.memo3               = 'gcm         : ' + ', '.join(gcms)
    srcs.memo4               = 'ghm         : ' + ', '.join(_ghms)

    times[:]      = Times
    latitudes[:]  = lats
    longitudes[:] = lons
    srcs[:]       = src

    rootgrp.close()
    print('\nFinished writting   : {outputpath} {src.shape} {src.min()}-{src.max()}\n')


# ----------------------------------------------------------------------------------------------------------------------
def drawmap_members(src, prd, srctype, outpath):

    src = ma.masked_equal(src, 0)

    norm = colors.Normalize()
    if srctype == 'abs':
        bounds = [0,10,20,30,40,50]
        cmap = cm.jet
    elif srctype == 'diff':
        bounds = [-60,-30,0,30,60]
        cmap = cm.bwr
    elif srctype == 'change_pc':
        bounds = [-10., 0., 10., 50., 100., 200]
        colors = divide([[ 0., 0.,180.],[0.,132.,132.],[240.,210., 0.],[230.,120., 0.],[170., 0., 0.]], 255.)
        cmap   = colors.ListedColormap(colors)
        norm   = colors.BoundaryNorm(bounds, cmap.N)

    if projection == 'cyl': ys = 11; ye = 293
    else:                   ys =  0; ye = 720

    fig = plt.figure(figsize=(15, 5.8))
    gs = gridspec.GridSpec(len(_ghms), len(gcms))  # (rows,cols)
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.5, top=0.95, wspace=0.01, hspace=0.005)
    plt.suptitle(f'{prd} {srctype}')

    for (j, gcm), (i, ghm) in itertools.product(enumerate(gcms), enumerate(_ghms)):

        ax = plt.subplot(gs[i,j])
        ax.axis('off')
        ax.set_title(f'{ghm} {gcm}', fontsize=8)
        im = bm.imshow(flipud(src[j,i,ys:ye,:]), norm=norm, cmap=cmap, vmin=bounds[0], vmax=bounds[-1], interpolation='nearest')
        bm.drawcoastlines(linewidth=0.2)

        if j == 0 and i == len(_ghms)-1:
            cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, boundaries=bounds, spacing='uniform', orientation='horizontal')
            cb.set_ticks(bounds)
            cb.set_ticklabels([str(int(i)) for i in bounds])

    plt.savefig(outpath)
    print('draw member map...'.format(outpath))
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def drawmap_s2n(src, prd, srctype, mmetype, figurepath):

    if mmetype == 'median':
        vmax = 3.5
    elif mmetype == 'mean':
        vmax = 2.5

    fig = plt.figure(figsize=(4, 1.7))
    ax = fig.add_subplot(111)
    plt.suptitle(f'{prd} {srctype}')
    im = plt.imshow(src, vmin=0, vmax=vmax)
    cb = plt.colorbar(im, orientation='horizontal', pad=0, aspect=35)
    plt.savefig(figurepath)
    print(f'drawmap_s2n... {figurepath}')
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def calc_area_frac(src, bounds, excelpath):

    _area = area[11:293]
    index = []
    values = []
    for bound in bounds:  # ex. [-200, -100, -50, 0, 50, 100, 200]
        if bound == 0:
            indexname = '<0'
            mask = np.ma.masked_greater_equal(src, 0).mask
            frac = np.ma.masked_array(_area, mask=mask).sum() / _area.sum()
            index.append(indexname); values.append(frac)
            indexname = '0<'
            mask = np.ma.masked_less_equal(src, 0).mask
            frac = np.ma.masked_array(_area, mask=mask).sum() / _area.sum()
            index.append(indexname); values.append(frac)
        elif bound < 0:
            indexname = '<{}'.format(bound)
            mask = np.ma.masked_greater(src, bound).mask
            frac = np.ma.masked_array(_area, mask=mask).sum() / _area.sum()
            index.append(indexname); values.append(frac)
        elif bound >0:
            indexname = '{}<'.format(bound)
            mask = np.ma.masked_less(src, bound).mask
            frac = np.ma.masked_array(_area, mask=mask).sum() / _area.sum()
            index.append(indexname); values.append(frac)

    df = pd.DataFrame(data=values, index=index, columns=['landarea_fraction'])
    writer = pd.ExcelWriter(excelpath)
    df.to_excel(writer)
    writer.save()
    writer.close()
    print(f'excel write: {excelpath}')


# ----------------------------------------------------------------------------------------------------------------------
def main(*args):

    strTIME = datetime.datetime.now()
    print('START basicMap.separate.output_climateindices.py\n@{}\n'.format(strTIME.strftime("%Y-%m-%d %H:%M:%S")))

    for scenario, ensemble_type, variable, distribution, scale, drought_severity, name in itertools.product(scenarios, ensemble_types, variables,
                                                                                                            distributions, scales, drought_severities, names
                                                                                                            ):
        strTime = datetime.datetime.now()
        print(f'\nJob started !!!  {scenario} {variable}')
        print(strTIME.strftime("%Y-%m-%d %H:%M:%S"))
        print('\n\n===================\n {} {} {} {} {} {} {}\n===================\n'.format(scenario, ensemble_type, variable, 
                                                                                    distribution, scale, drought_severity, name))
        if variable == 'pr': ghms = ['dummy']
        else:                ghms = _ghms
        nghm, ngcm = len(ghms), len(gcms)
        nmembers = nghm*ngcm

        # Reading data
        srcs = array([[read_netcdf(variable, name, drought_severity, scale, distribution, ghm, gcm, scenario) for ghm in ghms] for gcm in gcms])  #(ngcm,nghm,nperiod,nland)  _Filled_value = 1e+20
        srcs = ma.masked_equal(srcs, 1e+20)  # orignally, missing_value in each data is 1e+20
        print(f'srcs.shape: {srcs.shape}')  # (nperiod, ngcm, nghm, nland)
        srcs = np.transpose(srcs, axes=(2,0,1,3))
        print(f'srcs.shape: {srcs.shape}')  # (nperiod, ngcm, nghm, nland)

        # Kolmogorov-Smirnov test and dictKSmask
        # if p-value is less than ks, change in the grid is significant.
        dictKSmask = {period: {} for period in periods[1:]}
        dictSignif = {period: {} for period in periods[1:]}
        if KStest:
            print('\nKS testing...')
            for i, f in enumerate(periods[1:]):
                dictKSmask[f]['all'] = ma.make_mask(fillinMasked(array([0 if stats.ks_2samp(samples_hist.reshape(-1),samples_future.reshape(-1)).pvalue < ks else 1 for samples_hist, samples_future in zip(srcs[0].T, srcs[1:][i].T) ]), lnd_mask)==1) #(nY,nX)
                dictSignif[f]['all'] = ma.masked_array(area, mask=dictKSmask[f]['all']).sum() / area.sum()
                print(f' {f}       all', dictSignif[f]['all'])
                write2nc(dictKSmask[f]['all'], 'ksmask', scenario, f, ensemble_type, variable, distribution, scale, drought_severity)
                for j, ghm in enumerate(ghms):
                    dictKSmask[f][ghm] = ma.make_mask(fillinMasked(array([0 if stats.ks_2samp(samples_hist.reshape(-1),samples_future.reshape(-1)).pvalue < ks else 1 for samples_hist, samples_future in zip(srcs[0][:,j,...].T, srcs[1:][i,:,j,...].T)]), lnd_mask) == 1) #(nY,nX)
                    dictSignif[f][ghm] = ma.masked_array(area, mask=dictKSmask[f][ghm]).sum() / area.sum()
                    print(f' {f} {ghm}', dictSignif[f][ghm])
        else:
            print('KStest is OFF. Skip it.')
            for i, f in enumerate(periods[1:]):
                dictKSmask[f]['all'] = np.zeros((ny, nx), 'bool')
                for j, ghm in enumerate(ghms):
                    dictKSmask[f][ghm] = np.zeros((ny, nx), 'bool')

        # make climatology for each combination of GHM&GCM  (for each ensemble members)
        print('\ngenarating values...')
        climatology_base_period = srcs[0]   #         (ngcm,nghm,nLand)
        climatology_periods     = srcs[1:]  # (nperiod,ngcm,nghm,nLand)
        del srcs
        # percentage change (change_pc) for each ensemble members [%]
        climatological_diffs = climatology_periods - climatology_base_period                           # (nperiod,ngcm,nghm,nLand)
        climatological_diffs_percent = np.divide(climatological_diffs, climatology_base_period) * 100  # (nperiod,ngcm,nghm,nLand) [%]

        """
        # Just to make sure each members...
        if membermap:
            print('\nmember mapping...')

            figure_directory = genpath(figure_directory_main, scenario, 'member_map')
            if not os.path.isdir(figure_directory): os.makedirs(figure_directory)

            figure_name= '{}_abs_base.png'.format(variable)
            figurepath = genpath(figure_directory, figure_name)
            drawmap_members(ma.masked_equal(fillinMasked(climatology_base_period, lnd_mask), 1e+20), 'base', 'abs', figurepath)  # TODO: base?

            for figure_type, src in [('diffabs', climatological_diffs), ('diffpc', climatological_diffs_percent)]:
                for iperiod, period in enumerate(periods):

                    figure_name= '{}_{}_{}.png'.format(variable, figure_type, period)
                    figurepath = genpath(figure_directory, figure_name)
                    drawmap_members(ma.masked_equal(fillinMasked(src[iperiod], lnd_mask), 1e+20), period, figure_type, figurepath)
        """

        # mmed_climatological_diffs_percent : change >> ensembling
        if   ensemble_type == 'median':
            mmed_climatological_diffs_percent           = median(climatological_diffs_percent, axis=(1,2))  # (nPRC,          nLand) [%]
            ensembled_ghms_climatological_diffs_percent = median(climatological_diffs_percent, axis=1)      # (nPRC,     nghm,nLand) [%]
        elif ensemble_type == 'mean':
            mmed_climatological_diffs_percent           = mean(climatological_diffs_percent, axis=(1,2))    # (nPRC,          nLand) [%]
            ensembled_ghms_climatological_diffs_percent = mean(climatological_diffs_percent, axis=1)        # (nPRC,     nghm,nLand) [%]
        else:
            raise ValueError('Warning. Check ensemble_type.')

        # get periodical ensembled values
        if ensemble_type == 'median':
            fullensemble_base_period   = median(climatology_base_period, axis=(0,1))  #              (nLand)
            fullensemble_periods       = median(climatology_periods, axis=(1,2))      # (nperiod,     nLand)
            ensembled_ghms_base_period = median(climatology_base_period, axis=0)      # (        nghm,nLand)
            ensembled_ghms_periods     = median(climatology_periods, axis=1)          # (nperiod,nghm,nLand)
        elif ensemble_type == 'mean':
            fullensemble_base_period   = mean(climatology_base_period, axis=(0,1))    #              (nLand)
            fullensemble_periods       = mean(climatology_periods, axis=(1,2))        # (nperiod,        nLand)
            ensembled_ghms_base_period = mean(climatology_base_period, axis=0)        # (        nghm,nLand)
            ensembled_ghms_periods     = mean(climatology_periods, axis=1)            # (nperiod,   nghm,nLand)

        # Change (change_abs) & Percentage change (change_pc) in "ensembled" values [%]
        change_abs_fullensemble   = fullensemble_periods - fullensemble_base_period      # (nperiod,     nLand)
        change_abs_ensembled_ghms = ensembled_ghms_periods - ensembled_ghms_base_period  # (nperiod,nghm,nLand)
        # change_pc_fullensemble : ensemblig >> change
        change_pc_fullensemble   = np.divide(change_abs_fullensemble, fullensemble_base_period) * 100      # (nperiod,     nLand)
        change_pc_ensembled_ghms = np.divide(change_abs_ensembled_ghms, ensembled_ghms_base_period) * 100  # (nperiod,nghm,nLand)

        # Spread(spread) among all ensemble samples
        if ensemble_type == 'median':   # get inter quartile range (IQR)
            # spread in absolute value
            spread_in_climatology_base_period = subtract(*percentile(climatology_base_period, [75,25], axis=(0,1)))               # (        ngcm,nghm,nLand) >> (     nLand)
            spread_in_climatology_periods = subtract(*percentile(climatology_periods, [75,25], axis=(1,2)))                       # (nperiod,ngcm,nghm,nLand) >> (nperiod,nLand)
            # spread in percent change   [%]
            spread_in_climatological_diffs_percent  = subtract(*percentile(climatological_diffs_percent,  [75,25], axis=(1,2)))   # (nperiod,ngcm,nghm,nLand) >> (nperiod,nLand)
        elif ensemble_type == 'mean':   # get standard deviation (std)
            # spread in absolute value
            spread_in_climatology_base_period = climatology_base_period.std(axis=(0,1))                                           # (        ngcm,nghm,nLand) >> (     nLand)
            spread_in_climatology_periods = climatology_periods.std(axis=(1,2))                                                   # (nperiod,ngcm,nghm,nLand) >> (nperiod,nLand)
            # spread in percent change
            spread_in_climatological_diffs_percent = climatological_diffs_percent.std(axis=(1,2))                                 # (nperiod,ngcm,nghm,nLand) >> (nperiod,nLand)

        # Uncertainty among all ensemble samples
        if ensemble_type == 'median':  ##  Singal to noise (s2n)   (MME mean/median divided by its inter-quartile range.  ref: Prudhomme et al. 2014)
            # s2n for historical absolute value
            s2n_fullensemble_base_period = np.divide(np.absolute(fullensemble_base_period), spread_in_climatology_base_period)    # (        nLand) >> Fig
            s2n_fullensemble_periods     = np.divide(np.absolute(fullensemble_periods), spread_in_climatology_periods)            # (nperiod,nLand) >> Fig
            # s2n for change_pc
            s2n_mmed_climatological_diffs_percent = np.divide(np.absolute(spread_in_climatological_diffs_percent), mmed_climatological_diffs_percent)   # (nperiod,nLand) >> Fig
        elif ensemble_type == 'mean':
            # s2n for historical absolute value
            s2n_fullensemble_base_period = np.divide(np.absolute(spread_in_climatology_base_period), fullensemble_base_period)                          # (     nLand) >> Fig
            s2n_fullensemble_periods     = np.divide(np.absolute(spread_in_climatology_periods),     fullensemble_periods)                                       # (nperiod,nLand) >> Fig
            # s2n for change_pc
            s2n_mmed_climatological_diffs_percent = np.divide(np.absolute(mmed_climatological_diffs_percent), spread_in_climatological_diffs_percent)
        

        # Uncertainty comparison: GHM vs GCM
        # Ratio of GCM variation to total variation   (ref. Schewe et al. 2014)
        # GCM variation was computed across all gcms for each GHM individually and then averaged over all ghms
        ratGCMvar_climatology_base_period           = np.divide(var(climatology_base_period,      axis=0).mean(axis=0), var(climatology_base_period,      axis=(0,1)))    #      (nLand) >> Fig
        ratGCMvar_mmed_climatological_diffs_percent = np.divide(var(climatological_diffs_percent, axis=1).mean(axis=1), var(climatological_diffs_percent, axis=(1,2)))         # (nperiod,nLand) >> Fig
        del climatology_base_period, climatology_periods
        del spread_in_climatology_base_period#, spread_in_climatological_diffs_percent

#        # Agreement on the sign of change (increase/decrease)  (0-1)
#        # all member
        flug_fullensemble = where(mmed_climatological_diffs_percent>0,1,0) + where(mmed_climatological_diffs_percent<0,-1,0)                     # (          nperiod,nLand)  1 or -1 or 0
        flug_each_ensembled = (where(climatological_diffs_percent>0,1,0) + where(climatological_diffs_percent<0,-1,0)).transpose(1,2,0,3)          # (ngcm,nghm,nperiod,nLand)  1 or -1 or 0
        agreementALL = where(flug_each_ensembled==flug_fullensemble,1,0).sum(axis=(0,1)) / float(nmembers)   # (          nperiod,nLand)  0~1
#        #map_agreement_info(flug_fullensemble[1], flug_each_ensembled[:,:,1])

        # GCM ansemble for each GHM
        flug_fullensemble  = where(ensembled_ghms_climatological_diffs_percent>0,1,0) + where(ensembled_ghms_climatological_diffs_percent<0,-1,0)                #      (nperiod,nghm,nLand)
        flug_each_ensembled = (where(climatological_diffs_percent>0,1,0) + where(climatological_diffs_percent<0,-1,0)).transpose(1,0,2,3)          # (ngcm,nperiod,nghm,nLand)
        agreementGCM =  where(flug_each_ensembled==flug_fullensemble,1,0).sum(axis=0) / float(ngcm)       #      (nperiod,nghm,nLand)
        del flug_fullensemble, flug_each_ensembled

        # Convert nLand >> nY, nX                                (Note: missing_value is 1e+20)
        mmed_climatological_diffs_percent           = fillinMasked(mmed_climatological_diffs_percent,           lnd_mask)         # (nperiod,     nY,nX)
        ensembled_ghms_climatological_diffs_percent = fillinMasked(ensembled_ghms_climatological_diffs_percent, lnd_mask)         # (nperiod,nghm,nY,nX)
        fullensemble_base_period                    = fillinMasked(fullensemble_base_period,                    lnd_mask)         #           (nY,nX)
        ensembled_ghms_base_period                  = fillinMasked(ensembled_ghms_base_period,                  lnd_mask)         #      (nghm,nY,nX)
        s2n_fullensemble_base_period                = fillinMasked(s2n_fullensemble_base_period,                lnd_mask)         #           (nY,nX)
        s2n_fullensemble_periods                    = fillinMasked(s2n_fullensemble_periods,                    lnd_mask)         # (nperiod,     nY,nX)
        s2n_mmed_climatological_diffs_percent       = fillinMasked(s2n_mmed_climatological_diffs_percent,       lnd_mask)         # (nperiod,     nY,nX)
        ratGCMvar_climatology_base_period           = fillinMasked(ratGCMvar_climatology_base_period,           lnd_mask)         #           (nY,nX)
        ratGCMvar_mmed_climatological_diffs_percent = fillinMasked(ratGCMvar_mmed_climatological_diffs_percent, lnd_mask)         # (nPRC,     nY,nX)
        agreementALL                                = fillinMasked(agreementALL,                                lnd_mask)         # (nperiod,     nY,nX)
        agreementGCM                                = fillinMasked(agreementGCM,                                lnd_mask)         # (nperiod,nghm,nY,nX)
        fullensemble_periods                        = fillinMasked(fullensemble_periods,                        lnd_mask)         # (nperiod,     nY,nX)
        ensembled_ghms_periods                      = fillinMasked(ensembled_ghms_periods,                      lnd_mask)         # (nperiod,nghm,nY,nX)
        #change_abs_fullensemble = fillinMasked(change_abs_fullensemble_, lnd_mask)                       # (nperiod,     nY,nX)
        #change_abs_ensembled_ghms = fillinMasked(change_abs_ensembled_ghms_, lnd_mask)                       # (nperiod,nghm,nY,nX)
        #change_pc_fullensemble = fillinMasked(change_pc_fullensemble_, lnd_mask)                           # (nperiod,     nY,nX)
        #change_pc_ensembled_ghms = fillinMasked(change_pc_ensembled_ghms, lnd_mask)                            # (nperiod,nghm,nY,nX)

        ## report the area rate of significant increase
        ## mask out grids with drought decrease and grids with statistically insignificant changes
        #print 'reporting the area rate of significant increase :)'
        #for i, f in enumerate(['nearfuture', 'farfuture']):
        #    print 'in {}'.format(f)
        #    checkmask = ma.mask_or(ma.make_mask(mmed_climatological_diffs_percent[i]<=0), dictKSmask[f]['all'])
        #    print 'ensemble result: {:1%}'.format(ma.masked_array(area, checkmask).sum() / area.sum())
        #    for j, ghm in enumerate(ghms):
        #        checkmask = ma.mask_or(ma.make_mask(ensembled_ghms_climatological_diffs_percent[i,j]<=0), dictKSmask[f]['all'])
        #        print '{:>15}: {:1%}'.format(ghm, ma.masked_array(area, checkmask).sum() / area.sum())
        print('mmed_climatological_diffs_percent.shape           {}'.format(mmed_climatological_diffs_percent.shape))
        print('agreementALL.shape                                {}'.format(agreementALL.shape))
        print('s2n_mmed_climatological_diffs_percent.shape       {}'.format(s2n_mmed_climatological_diffs_percent.shape))
        print('ensembled_ghms_climatological_diffs_percent.shape {}'.format(ensembled_ghms_climatological_diffs_percent.shape))
        print('agreementGCM.shape                                {}'.format(agreementGCM.shape))
        print('ratGCMvar_mmed_climatological_diffs_percent.shape {}'.format(ratGCMvar_mmed_climatological_diffs_percent.shape))

        # Make figure
        print('\n\n\nFigure making...')
        for prd,            figure_type,         (ensembled_src,                        agreementall),     s2n_src,                                  (ghm_srcs,                                       agreementgcm), uncertainty_source_rate in [
            ['base_period', 'absolute',          (fullensemble_base_period,             None),             s2n_fullensemble_base_period,             (ensembled_ghms_base_period,                     None), ratGCMvar_climatology_base_period],
            # ---
            #['recent30yrs', 'absolute',          (fullensemble_periods[0],              None),             s2n_fullensemble_periods[0],              (ensembled_ghms_periods[0],                      None), None],
            ['nearfuture',  'absolute',          (fullensemble_periods[1],              None),             s2n_fullensemble_periods[1],              (ensembled_ghms_periods[1],                      None), None],
            ['farfuture',   'absolute',          (fullensemble_periods[2],              None),             s2n_fullensemble_periods[2],              (ensembled_ghms_periods[2],                      None), None],
            # ---
            #['recent30yrs', 'percentage_change', (mmed_climatological_diffs_percent[0], agreementALL[0]),  s2n_mmed_climatological_diffs_percent[0], (ensembled_ghms_climatological_diffs_percent[0], agreementGCM[0]), ratGCMvar_mmed_climatological_diffs_percent[0]],
            ['nearfuture',  'percentage_change', (mmed_climatological_diffs_percent[1], agreementALL[1]),  s2n_mmed_climatological_diffs_percent[1], (ensembled_ghms_climatological_diffs_percent[1], agreementGCM[1]), ratGCMvar_mmed_climatological_diffs_percent[1]],
            ['farfuture',   'percentage_change', (mmed_climatological_diffs_percent[2], agreementALL[2]),  s2n_mmed_climatological_diffs_percent[2], (ensembled_ghms_climatological_diffs_percent[2], agreementGCM[2]), ratGCMvar_mmed_climatological_diffs_percent[2]],
            ###['1st-half-21C', 'percentage_change', (mmed_climatological_diffs_percent[0], agreementALL[0]),  s2n_mmed_climatological_diffs_percent[0], (ensembled_ghms_climatological_diffs_percent[0], agreementGCM[0]), ratGCMvar_mmed_climatological_diffs_percent[0]],
            ###['2nd-half-21C',  'percentage_change', (mmed_climatological_diffs_percent[1], agreementALL[1]),  s2n_mmed_climatological_diffs_percent[1], (ensembled_ghms_climatological_diffs_percent[1], agreementGCM[1]), ratGCMvar_mmed_climatological_diffs_percent[1]],
            ]:
            print(f'\n==========\n {prd} {figure_type}\n ==========')
            ensembled_src = ma.masked_equal(ensembled_src, 1e+20)
            s2n_src = ma.masked_equal(s2n_src, 1e+20)
            ghm_srcs = ma.masked_equal(ghm_srcs, 1e+20)
            print(f'ensembled_src: {ensembled_src.shape}')
            print(f's2n_src      : {s2n_src.shape}')
            print(f'ghm_srcs     : {ghm_srcs.shape}')
            if uncertainty_source_rate is not None: 
                uncertainty_source_rate = ma.masked_equal(uncertainty_source_rate, 1e+20)
                print('uncertainty_source_rate     : {} - {}   {}'.format(uncertainty_source_rate.min(), uncertainty_source_rate.max(), uncertainty_source_rate.shape))

            ## Just to check s2n value
            #if s2nmap:
            #    figurename  = 's2n_Q%02iwin%02i_Len%03itau%i_%s_%s_%s_%s.png'%(Q, win, Len, tau, season, variable, prd, figure_type)
            #    figure_directory   = genpath(figure_directory_main, soc, '%s.1971_2004'%(qvalType), scenario, ensemble_type)
            #    figurepath  = genpath(figure_directory,figurename)
            #    if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
            #    drawmap_s2n(s2n_src, prd, figure_type, ensemble_type, figurepath)

            if EnsmblMap and not figure_type == 'absolute':
                write2nc(ensembled_src, figure_type, scenario, prd, ensemble_type, variable, distribution, scale, drought_severity)  # write out data for other analyses...
            if agreementall is not None:
                write2nc(agreementall, 'agreement', scenario, prd, ensemble_type, variable, distribution, scale, drought_severity)  # write out data for other analyses...

            if   projection == 'cyl': 
                ensembled_src = ensembled_src[11:293,:]
                s2n_src = s2n_src[11:293,:]
                ghm_srcs = ghm_srcs[:,11:293,:]
                if not figure_type == 'absolute':
                    agreementall = agreementall[11:293,:]
                    agreementgcm = agreementgcm[:,11:293,:]
                    ksmask_all = dictKSmask[prd]['all'][11:293,:]
            elif projection == 'robin' or projection == 'eck4': 
                if figure_type == 'absolute':      pass
                else: ksmask_all = dictKSmask[prd]['all']
            else:
                raise ValueError('check projection')

            # Ensemble result ------------------------------------------------------------------------------------------
            if EnsmblMap:

                fig1 = plt.figure(num=1, figsize=(4, 1.7))
                gs = gridspec.GridSpec(1, 1)  # (rows,cols)
                gs.update(left=0.01, right=0.99, bottom=0.02, top=0.98, hspace=0.02, wspace=0.01)
                # ax1 (Upper left: Main ensemble value)
                ax1 = plt.subplot(gs[0,0])
                ax1.axis('off')
                ax_pos = ax1.get_position()
                norm1 = colors.Normalize()

                if not figure_type == 'absolute':
                    ensembled_src = ma.masked_array(ma.masked_equal(ensembled_src,0), mask=ksmask_all)

                ax1.text(0, 0.98, '{}, {}'.format(dict_standerdizedname[variable], scenario), ha='left', va='top', fontsize=6, transform=ax1.transAxes)

                if figure_type == 'percentage_change':
                    bounds = [-200, -100, -10., 0., 10., 100., 200.]
                    if len(agreeThrsh) == 2:
                        # ex) agreeThrsh = [0.8, 0.6]
                        mask1  = ma.make_mask(agreementall<agreeThrsh[0])
                        mask21 = ma.make_mask(agreementall>=agreeThrsh[0])
                        mask22 = ma.make_mask(agreementall<agreeThrsh[1])
                        mask2  = ma.mask_or(mask21, mask22)
                        mask3  = ma.make_mask(agreementall>=agreeThrsh[1])
                        signal1 = ma.masked_array(ma.masked_equal(ensembled_src,0), mask=mask1)
                        signal2 = ma.masked_array(ma.masked_equal(ensembled_src,0), mask=mask2)
                        signal3 = ma.masked_array(ma.masked_equal(ensembled_src,0), mask=mask3)
                        colors1 = divide([[  0.,  0.,204.],[  0.,102.,204.],[  0.,204.,204.],[204.,204.,  0.],[204.,102.,  0.],[204.,  0.,  0.]], 255.)
                        colors2 = divide([[153.,153.,255.],[153.,204.,255.],[153.,255.,255.],[255.,255.,153.],[255.,204.,153.],[255.,153.,153.]], 255.)
                        colors3 = divide([[230.,230.,230.],[230.,230.,230.],[230.,230.,230.],[230.,230.,230.],[230.,230.,230.],[230.,230.,230.]], 255.)  #gray...
                        cmap1 = colors.ListedColormap(colors1)
                        cmap2 = colors.ListedColormap(colors2)
                        cmap3 = colors.ListedColormap(colors3)
                        ims3 = bm.imshow(flipud(signal3), norm=norm1, cmap=cmap3, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                        ims2 = bm.imshow(flipud(signal2), norm=norm1, cmap=cmap2, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                        ims1 = bm.imshow(flipud(signal1), norm=norm1, cmap=cmap1, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                        bm.drawcoastlines(linewidth=0.2)
                        # 2D colorbar for Kaye et al.-plot:
                        if variable == 'qtot' and scenario == 'rcp26':
                            linewidth = 0.08
                            length = 8.3
                            fontsize = 4.5
                            ax11 = fig1.add_axes([ax_pos.x0+0.033, ax_pos.y0+0.130, 0.23, 0.07])
                            ax12 = fig1.add_axes([ax_pos.x0+0.033, ax_pos.y0+0.200, 0.23, 0.07])
                            ax13 = fig1.add_axes([ax_pos.x0+0.033, ax_pos.y0+0.270, 0.23, 0.07])
                            cmap = [cmap1, cmap2, cmap3]
                            axes = [ax12, ax13]
                            for i, axs in enumerate(axes):
                                norm = colors.BoundaryNorm(bounds, cmap[i+1].N)
                                cb = mpl.colorbar.ColorbarBase(axs, cmap=cmap[i+1], norm=norm, boundaries=bounds, spacing='uniform', orientation='horizontal')
                                cb.set_ticks(bounds)
                                cb.set_ticklabels([])
                                cb.ax.tick_params(direction='in', width=linewidth, length=length)
                                for axis in ['top','bottom','left','right']: axs.spines[axis].set_linewidth(linewidth)
                                cb.outline.set_visible(False)
                            norm = colors.BoundaryNorm(bounds, cmap[0].N)
                            cb1 = mpl.colorbar.ColorbarBase(ax11, cmap=cmap[0], norm=norm, boundaries=bounds, spacing='uniform', orientation='horizontal')
                            cb1.set_ticks(bounds)
                            cb1.set_ticklabels([str(int(i)) for i in bounds])
                            cb1.ax.tick_params(labelsize=fontsize, direction='in', width=linewidth, length=length, pad=1, rotation=45)
                            for axis in ['top','bottom','left','right']: ax11.spines[axis].set_linewidth(linewidth)
                            cb1.outline.set_visible(False)
                            cb1.set_label('relative change [%]', fontsize=fontsize, labelpad=-1.7)
                            fig1.text(ax_pos.x0+0.013,  ax_pos.y0+0.27, str(int(agreeThrsh[1]*1e2)), ha='left', va='center', fontsize=fontsize)
                            fig1.text(ax_pos.x0+0.013,  ax_pos.y0+0.20, str(int(agreeThrsh[0]*1e2)), ha='left', va='center', fontsize=fontsize)
                            fig1.text(ax_pos.x0,       ax_pos.y0+0.24, 'agreement [%]',             ha='left', va='center', fontsize=fontsize, rotation='vertical')
                    elif len(agreeThrsh) == 3:  # ex) agreeThrsh = [ 0.8, 0.7, 0.6 ]
                        mask1  = ma.make_mask(agreementall<agreeThrsh[0])
                        mask21 = ma.make_mask(agreementall>=agreeThrsh[0])
                        mask22 = ma.make_mask(agreementall<agreeThrsh[1])
                        mask2  = ma.mask_or(mask21, mask22)
                        mask31 = ma.make_mask(agreementall>=agreeThrsh[1])
                        mask32 = ma.make_mask(agreementall<agreeThrsh[2])
                        mask3  = ma.mask_or(mask31, mask32)
                        mask4  = ma.make_mask(agreementall>=agreeThrsh[2])
                        signal1 = ma.masked_array(ma.masked_equal(ensembled_src,0), mask=mask1)
                        signal2 = ma.masked_array(ma.masked_equal(ensembled_src,0), mask=mask2)
                        signal3 = ma.masked_array(ma.masked_equal(ensembled_src,0), mask=mask3)
                        signal4 = ma.masked_array(ma.masked_equal(ensembled_src,0), mask=mask4)
                        colors1 = divide([[  0.,  0.,204.],[  0.,102.,204.],[  0.,204.,204.],[204.,204.,  0.],[204.,102.,  0.],[204.,  0.,  0.]], 255.)
                        colors2 = divide([[ 51., 51.,255.],[ 51.,153.,255.],[ 51.,255.,255.],[255.,255., 51.],[255.,153., 51.],[255., 51., 51.]], 255.)
                        colors3 = divide([[153.,153.,255.],[153.,204.,255.],[153.,255.,255.],[255.,255.,153.],[255.,204.,153.],[255.,153.,153.]], 255.)
                        colors4 = divide([[230.,230.,230.],[230.,230.,230.],[230.,230.,230.],[230.,230.,230.],[230.,230.,230.],[230.,230.,230.]], 255.)  #gray...
                        cmap1 = colors.ListedColormap(colors1)
                        cmap2 = colors.ListedColormap(colors2)
                        cmap3 = colors.ListedColormap(colors3)
                        cmap4 = colors.ListedColormap(colors4)
                        ims3 = bm.imshow(flipud(signal3), norm=norm1, cmap=cmap3, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                        ims2 = bm.imshow(flipud(signal2), norm=norm1, cmap=cmap2, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                        ims1 = bm.imshow(flipud(signal1), norm=norm1, cmap=cmap1, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                        ims4 = bm.imshow(flipud(signal4), norm=norm1, cmap=cmap4, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                        bm.drawcountries(linewidth=0.03)
                        bm.drawcoastlines(linewidth=0.2)
                        # 2D colorbar for Kaye et al.-plot:
                        ax11 = fig1.add_axes([ax_pos.x0+0.025+0.05, ax_pos.y0+0.130, 0.15, 0.04])
                        ax12 = fig1.add_axes([ax_pos.x0+0.025+0.05, ax_pos.y0+0.170, 0.15, 0.04])
                        ax13 = fig1.add_axes([ax_pos.x0+0.025+0.05, ax_pos.y0+0.210, 0.15, 0.04])
                        ax14 = fig1.add_axes([ax_pos.x0+0.025+0.05, ax_pos.y0+0.250, 0.15, 0.04])
                        cmap = [cmap1, cmap2, cmap3, cmap4]
                        for i, axs in enumerate([ax12, ax13, ax14]):
                          norm = colors.BoundaryNorm(bounds, cmap[i+1].N)
                          cb = mpl.colorbar.ColorbarBase(axs, cmap=cmap[i+1], norm=norm, boundaries=bounds, spacing='uniform', orientation='horizontal')
                          cb.set_ticks(bounds)
                          cb.set_ticklabels([])
                          cb.ax.tick_params(labelsize=3,direction='in')
                        norm = colors.BoundaryNorm(bounds, cmap[0].N)
                        cb1 = mpl.colorbar.ColorbarBase(ax11, cmap=cmap[0], norm=norm, boundaries=bounds, spacing='uniform', orientation='horizontal')
                        cb1.set_ticks(bounds)
                        cb1.set_ticklabels([str(int(i)) for i in bounds])
                        cb1.ax.tick_params(labelsize=3, direction='in')
                        cb1.set_label('relative change [%s]', fontsize=3)#, labelpad=-0.6)
                        fig1.text(ax_pos.x0+0.016+0.05, ax_pos.y0+0.25, str(int(agreeThrsh[2]*1e2)), va='center', ha='center', fontsize=3)
                        fig1.text(ax_pos.x0+0.016+0.05, ax_pos.y0+0.21, str(int(agreeThrsh[1]*1e2)), va='center', ha='center', fontsize=3)
                        fig1.text(ax_pos.x0+0.016+0.05, ax_pos.y0+0.17, str(int(agreeThrsh[0]*1e2)), va='center', ha='center', fontsize=3)
                        fig1.text(ax_pos.x0+0.008+0.05, ax_pos.y0+0.20, 'agreement [%]', va='center', ha='center', rotation='vertical', fontsize=3)

                else:   # historical absolute or absolute change
                    percentile_value = np.round(np.percentile(np.abs(ensembled_src.compressed()), 90), 4)
                    if figure_type == 'absolute_change':
                        bounds = [-percentile_value, 0, percentile_value]
                        cmap = cm.bwr
                    elif figure_type == 'absolute':
                        bounds = [0, percentile_value]
                        cmap = cm.hot_r

                    ims1 = bm.imshow(flipud(ensembled_src), norm=norm1, cmap=cmap, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                    bm.drawcoastlines(linewidth=0.2)

                    ax11 = fig1.add_axes([ax_pos.x0+0.025+0.05, ax_pos.y0+0.160,  0.15, 0.03])
                    cb1 = mpl.colorbar.ColorbarBase(ax11, cmap=cmap, norm=norm1, orientation='horizontal')
                    cb1.set_ticks(bounds)
                    cb1.set_ticklabels([str(int(i)) for i in bounds])
                    cb1.ax.tick_params(labelsize=3,direction='in')
                    if figure_type == 'absolute_change':
                        cb1.set_label('{} [{}]'.format(figure_type,dict_unit[name][0]), fontsize=3)#, labelpad=-0.6)
                    elif figure_type == 'absolute':
                        cb1.set_label('[{}]'.format(dict_unit[name][0]), fontsize=3)#, labelpad=-0.6)

                ## KS test significance
                #if (prd == 'nearfuture' or prd == 'farfuture') and (figure_type == 'absolute_change' or figure_type == 'percentage_change'):
                #    if KStest: ax1.text(0.95,0.02, 'significant change over %.1f%% of globe land area with KS test'%(dictSignif[f]['all']*100),\
                #                        va="bottom", ha="right", fontsize=3, transform=ax1.transAxes)
                #ax1.text(0.01, 0.98, '(a)', va="top", ha="left", fontsize=8, transform=ax1.transAxes)

                figurename = f'basicMap.ensemble{ensemble_type}_{name}_{drought_severity}_{variable}_{prd}_{figure_type}.'
                figure_directory = genpath(figure_directory_main, scenario, ensemble_type, variable, distribution, str(scale))
                if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
                figurepath = genpath(figure_directory, figurename)
                for suffix in suffixes:
                    if os.path.exists(figurepath+suffix):
                        print('File exists, will be overwritten.')
                    plt.savefig(figurepath+suffix, dpi=savefig_dpi)
                    print('savefig: {}'.format(figurepath + suffix))
                plt.close(1)

                if figure_type == 'percentage_change':
                    excelname = f'area_frac_{scenario}_{prd}_{ensemble_type}_{variable}_{distribution}_{scale}_{drought_severity}.xlsx'
                    excelpath = genpath(figure_directory, excelname)
                    calc_area_frac(ensembled_src, bounds, excelpath)

            # signal to noize ------------------------------------------------------------------------------------------
            if s2nmap:
                fig1 = plt.figure(num=2, figsize=(4, 1.7))
                gs = gridspec.GridSpec(1, 1)  # (rows,cols)
                gs.update(left=0.01, right=0.99, bottom=0.02, top=0.98, hspace=0.02, wspace=0.01)
                ax2 = plt.subplot(gs[0,0])
                ax2.axis('off')
                ax_pos = ax2.get_position()
                norm1 = colors.Normalize()
                ax2.text(0, 0.98, '{}, {}'.format(dict_standerdizedname[variable], scenario), ha='left', va='top', fontsize=6, transform=ax1.transAxes)

                if   ensemble_type == 'median':
                    bounds = [0,0.5,1,1.5,2,2.5,3,3.5]
                    colors1 = divide([[0,0,180], [28,125,199], [238,172,172], [228,121,121], [239,0,0], [198,0,0], [158,0,0]], 255.)
                    labelName = 'signal to noize ratio [-]'
                elif ensemble_type == 'mean'  :
                    bounds = [0,0.05,0.1,0.5,1,1.5,2,2.5]
                    colors1 = divide([[0,0,180], [28,125,199], [238,172,172], [228,121,121], [239,0,0], [198,0,0], [158,0,0]], 255.)
                    labelName = 'coefficient of variation [-]'
                cmap = colors.ListedColormap(colors1)
                norm = colors.BoundaryNorm(bounds, cmap.N)
                im = bm.imshow(flipud(s2n_src), norm=norm, cmap=cmap, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                bm.drawcoastlines(linewidth=0.2)
                bm.drawmapboundary(fill_color='aqua', linewidth=0.01)
                ax11 = fig1.add_axes([ax_pos.x0+0.02, ax_pos.y0+0.16,  0.10, 0.03])
                cb1 = mpl.colorbar.ColorbarBase(ax11,
                                                cmap=cmap,
                                                norm=norm,
                                                boundaries=bounds,
                                                spacing='uniform',
                                                orientation='horizontal')
                cb1.set_ticks(bounds)
                cb1.set_ticklabels([str(i) for i in bounds])
                cb1.ax.tick_params(labelsize=6,direction='in')
                cb1.set_label(labelName, fontsize=7)#, labelpad=-0.6)
                #ax2.text(0.01, 0.98, '(b)', va="top", ha="left", fontsize=8, transform=ax2.transAxes)

                # add information
                #if figure_type == 'absolute_change' or figure_type == 'percentage_change':
                #    ax2.text(0.38, 0.01,
                #             #'%s\n(Simulation: %s, %s (Period: %s-historical %s (Drought: Q%i, Len%i'%(figure_type,scenario,soc,prd,season,Q,Len),
                #             '%s\n%s, %s,  %s-historical %s,  Q%i, Len%i'%(figure_type, scenario, soc, prd, season, Q, Len),
                #             va='bottom', ha='left', fontsize=5, transform=ax2.transAxes)
                #elif figure_type == 'absolute':
                #    ax2.text(0.38, 0.01,
                #             #'%s\n(Simulation: %s, %s  (Period: %s %s  (Drought: Q%i, Len%i'%(figure_type,scenario,soc,prd,season,Q,Len),
                #             '%s\n%s, %s,  %s %s,  Q%i, Len%i'%(figure_type, scenario, soc, prd, season, Q, Len),
                #             va='bottom', ha='left', fontsize=5, transform=ax2.transAxes)

                figurename = f'basicMap.s2n_{ensemble_type}_{name}_{drought_severity}_{variable}_{prd}_{figure_type}.'
                figure_directory = genpath(figure_directory_main, scenario, ensemble_type, variable, distribution, str(scale))
                if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
                figurepath = genpath(figure_directory, figurename)

                for suffix in suffixes:
                    if os.path.exists(figurepath+suffix):
                        print('File exists, will be overwritten.')
                    plt.savefig(figurepath+suffix, dpi=savefig_dpi)
                    print('savefig: {}'.format(figurepath + suffix))
                plt.close(2)


            # each GHM -------------------------------------------------------------------------------------------------
            if ghmsMap:

                space = 0.005
                map_width = 0.32828
                map_height = 0.27

                fig2 = plt.figure(num=3, figsize=(12, 7.5))
                ax1 = fig2.add_axes([space,                0.999-map_height*1-0.001*9, map_width, map_height])
                ax2 = fig2.add_axes([space*1.5+map_width,  0.999-map_height*1-0.001*9, map_width, map_height])
                ax3 = fig2.add_axes([space*3+map_width*2,  0.999-map_height*1-0.001*9, map_width, map_height])
                ax4 = fig2.add_axes([space,                0.999-map_height*2-0.001*9, map_width, map_height])
                ax5 = fig2.add_axes([space*1.5+map_width,  0.999-map_height*2-0.001*9, map_width, map_height])
                ax6 = fig2.add_axes([space*3+map_width*2,  0.999-map_height*2-0.001*9, map_width, map_height])
                ax7 = fig2.add_axes([space,                0.999-map_height*3-0.001*9, map_width, map_height])
                ax8 = fig2.add_axes([space*1.5+map_width,  0.999-map_height*3-0.001*9, map_width, map_height])

                percentile_value = np.round(np.percentile(np.abs(ghm_srcs.compressed()), 90), 4)
                for i, (j, ghm, ax) in enumerate(zip(['a','b','c', 'd', 'e', 'f', 'g', 'h'],
                                                      ghms,
                                                      [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8])):
                    print('ax{}...'.format(i+1))
                    plt.sca(ax)
                    ax.axis('off')
                    ax_pos = ax.get_position()

                    if not figure_type == 'absolute':
                        if   projection == 'cyl':                           ks_mask = dictKSmask[prd][ghm][11:293]
                        elif projection == 'robin' or projection == 'eck4': ks_mask = dictKSmask[prd][ghm]
                        else: raise ValueError('check projection')

                    aSrc = ghm_srcs[i]
                    if not figure_type == 'absolute':
                        aSrc = ma.masked_array(ma.masked_equal(aSrc, 0), mask=ks_mask)

                    if figure_type == 'percentage_change':

                        agreement = agreementgcm[i]
                        agreement = ma.masked_equal(agreement, 1e+20)
                        mask1   = ma.make_mask(agreement<agreeThrsh[0])
                        mask21  = ma.make_mask(agreement>=agreeThrsh[0])
                        mask22  = ma.make_mask(agreement<agreeThrsh[1])
                        mask2   = ma.mask_or(mask21, mask22)
                        mask31  = ma.make_mask(agreement>=agreeThrsh[1])
                        mask32  = ma.make_mask(agreement<agreeThrsh[2])
                        mask3   = ma.mask_or(mask31, mask32)
                        mask4   = ma.make_mask(agreement>=agreeThrsh[2])
                        signal1 = ma.masked_array(ma.masked_equal(aSrc,0), mask=mask1)
                        signal2 = ma.masked_array(ma.masked_equal(aSrc,0), mask=mask2)
                        signal3 = ma.masked_array(ma.masked_equal(aSrc,0), mask=mask3)
                        signal4 = ma.masked_array(ma.masked_equal(aSrc,0), mask=mask4)

                        bounds = [-200, -100, -50., 0., 50., 100., 200.]
                        colors1 = divide([[  0.,  0.,204.],[  0.,102.,204.],[  0.,204.,204.],[204.,204.,  0.],[204.,102.,  0.],[204.,  0.,  0.]], 255.)
                        colors2 = divide([[ 51., 51.,255.],[ 51.,153.,255.],[ 51.,255.,255.],[255.,255., 51.],[255.,153., 51.],[255., 51., 51.]], 255.)  
                        colors3 = divide([[153.,153.,255.],[153.,204.,255.],[153.,255.,255.],[255.,255.,153.],[255.,204.,153.],[255.,153.,153.]], 255.)  
                        colors4 = divide([[230.,230.,230.],[230.,230.,230.],[230.,230.,230.],[230.,230.,230.],[230.,230.,230.],[230.,230.,230.]], 255.)  #gray...
                        cmap1 = colors.ListedColormap(colors1)
                        cmap2 = colors.ListedColormap(colors2)
                        cmap3 = colors.ListedColormap(colors3)
                        cmap4 = colors.ListedColormap(colors4)
                        cmaps = [cmap1, cmap2, cmap3, cmap4]
                        ims3 = bm.imshow(flipud(signal3), norm=norm1, cmap=cmap3, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest");
                        ims2 = bm.imshow(flipud(signal2), norm=norm1, cmap=cmap2, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest");
                        ims1 = bm.imshow(flipud(signal1), norm=norm1, cmap=cmap1, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest");
                        ims4 = bm.imshow(flipud(signal4), norm=norm1, cmap=cmap4, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest");
                        bm.drawcoastlines(linewidth=0.2)

                    else:   # historical absolute or change

                        if figure_type == 'absolute_change':
                            bounds = [-percentile_value, 0, percentile_value]
                            cmap = cm.bwr
                        elif figure_type == 'absolute':
                            bounds = [0, percentile_value]
                            cmap = cm.hot_r

                        ims1 = bm.imshow(flipud(aSrc),norm=norm1,cmap=cmap,vmin=bounds[0],vmax=bounds[-1],interpolation="nearest")
                        bm.drawcoastlines(linewidth=0.2)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # KS test significance
                        #if (prd == 'nearfuture' or prd == 'farfuture') and (figure_type == 'absolute_change' or figure_type == 'percentage_change'):
                        #    if j == '':
                        #        if KStest: ax.text(0.37,0.01, 'significant change\nover %.1f%% of globe land area with KS test'%(dictSignif[f][ghm]*100),
                        #                            va="bottom", ha="left", fontsize=7, transform=ax.transAxes)
                        #    else:
                        #        if KStest: ax.text(0.95,0.01, 'significant change over %.1f%%'%(dictSignif[f][ghm]*100),
                        #                            va="bottom", ha="right", fontsize=7, transform=ax.transAxes)
                    """
                    # add information:
                    if j == 'd':
                        if figure_type == 'absolute_change' or figure_type == 'percentage_change':
                            fig2.text(ax_pos.x0+0.015, ax_pos.y0-0.04, 
                                      '%s\n\n Simulation:\n - %s\n - %s\nPeriod:\n - %s v.s. historical\n - %s\nDrought:\n - Q%i\n - Len%i'%(
                                      figure_type,scenario,soc,prd,season,Q,Len),
                                      va='top', ha='left', fontsize=7)
                        elif figure_type == 'absolute':
                            fig2.text(ax_pos.x0+0.015, ax_pos.y0-0.04, 
                                      '%s\n\n Simulation:\n - %s\n - %s\nPeriod:\n - %s\n - %s\nDrought:\n - Q%i\n - Len%i'%(
                                      figure_type,scenario,soc,prd,season,Q,Len),
                                      va='top', ha='left', fontsize=7)
                    """
                    # add subplot number
                    ax.text(0.01, 0.02, f'({j}) {ghm}', va='bottom', ha='left', fontsize=8, transform=ax.transAxes)

                # add a common colorbar 
                width = 0.17
                hight = 0.03

                if figure_type == 'percentage_change':

                    # 2D colorbar for Kaye et al.-plot:
                    bounds = [-200, -100, -50, 0., 10., 100., 200]
                    ax_pos = ax8.get_position()
                    ax11 = fig2.add_axes([ax_pos.x0+width+0.25, ax_pos.y0+0.00+hight, width, hight])  # lowest one
                    ax12 = fig2.add_axes([ax_pos.x0+width+0.25, ax_pos.y0+0.03+hight, width, hight])
                    ax13 = fig2.add_axes([ax_pos.x0+width+0.25, ax_pos.y0+0.06+hight, width, hight])
                    ax14 = fig2.add_axes([ax_pos.x0+width+0.25, ax_pos.y0+0.09+hight, width, hight])

                    for i, axs in enumerate([ax12, ax13, ax14]):
                      norm = colors.BoundaryNorm(bounds, cmaps[i+1].N)
                      cb = mpl.colorbar.ColorbarBase(axs, cmap=cmaps[i+1], norm=norm,
                                                     boundaries=bounds,
                                                     #extend='neither',
                                                     #extendfrac='auto',
                                                     #spacing='proportional',
                                                     spacing='uniform',
                                                     orientation='horizontal')
                      cb.set_ticks(bounds)
                      cb.set_ticklabels([])
                      cb.ax.tick_params(labelsize=8,direction='in')
                    norm = colors.BoundaryNorm(bounds, cmaps[0].N)
                    cb1 = mpl.colorbar.ColorbarBase(ax11, cmap=cmaps[0], norm=norm,
                                                    boundaries=bounds,
                                                    #extend='neither',
                                                    #extendfrac='auto',
                                                    #spacing='proportional',
                                                    spacing='uniform',
                                                    orientation='horizontal')
                    cb1.set_ticks(bounds)
                    cb1.set_ticklabels([str(int(i)) for i in bounds])
                    cb1.ax.tick_params(labelsize=8,direction='in')
                    cb1.set_label('relative change [%]', fontsize=8)#, labelpad=-0.60)
                    fig2.text(ax_pos.x0+width+0.245, ax_pos.y0+0.09+hight, str(int(agreeThrsh[2]*1e2)), va='center', ha='center', fontsize=7)
                    fig2.text(ax_pos.x0+width+0.245, ax_pos.y0+0.06+hight, str(int(agreeThrsh[1]*1e2)), va='center', ha='center', fontsize=7)
                    fig2.text(ax_pos.x0+width+0.245, ax_pos.y0+0.03+hight, str(int(agreeThrsh[0]*1e2)), va='center', ha='center', fontsize=7)
                    fig2.text(ax_pos.x0+width+0.220, ax_pos.y0+0.06+hight, 'agreement [%]',             va='center', ha='center', fontsize=8, rotation='vertical')

                else:

                    ax_pos = ax8.get_position()
                    ax11 = fig2.add_axes([ax_pos.x0+width+0.25, ax_pos.y0+hight, width, hight])
                    cb1 = mpl.colorbar.ColorbarBase(ax11, cmap=cmap, norm=norm1, orientation='horizontal')
                    cb1.set_ticks(bounds)
                    cb1.set_ticklabels([str(int(i)) for i in bounds])
                    cb1.ax.tick_params(labelsize=8,direction='in')
                    if figure_type == 'absolute_change':
                        cb1.set_label('{} [{}]'.format(figure_type,dict_unit[name][0]), fontsize=8)#, labelpad=-0.6)
                    elif figure_type == 'absolute':
                        cb1.set_label('[{}]'.format(dict_unit[name][0]), fontsize=8)#, labelpad=-0.6)

                figure_directory = genpath(figure_directory_main, scenario, ensemble_type, variable, distribution, str(scale))
                if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
                figurename = f'basicMap{ensemble_type}.ghms_{name}_{drought_severity}_{variable}_{prd}_{figure_type}.'
                figurepath  = genpath(figure_directory, figurename)

                for suffix in suffixes:
                    if os.path.exists(figurepath+suffix):
                        print('File exists, will be overwritten.')
                    plt.savefig(figurepath+suffix, dpi=savefig_dpi)
                    print('savefig: {}'.format(figurepath+suffix))
                plt.close(3)

            # 4. Uncertainty source ------------------------------------------------------------------------------------
            if UncertaintySource and uncertainty_source_rate is not None:

                aSrc = uncertainty_source_rate[11:293,:] * 100
                bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # [%]
                norm = colors.Normalize()

                fig1 = plt.figure(num=1, figsize=(4, 1.7))
                gs = gridspec.GridSpec(1, 1)  # (rows,cols)
                gs.update(left=0.01, right=0.99, bottom=0.02, top=0.98, hspace=0.02, wspace=0.01)
                ax1 = plt.subplot(gs[0,0])
                ax1.axis('off')
                im1 = bm.imshow(flipud(aSrc), norm=norm, cmap=cm.RdYlBu, vmin=bounds[0], vmax=bounds[-1], interpolation="nearest")
                bm.drawcoastlines(linewidth=0.2)
                #ax1.text(0.5, 1., '{} {}'.format(scn,future), va="bottom", ha="center", fontsize=8, transform=ax.transAxes)
                ax_pos = ax1.get_position()
                ax2 = fig1.add_axes([ax_pos.x0+0.5, ax_pos.y0+0.055,  0.425, 0.02])
                cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cm.RdYlBu, norm=norm, orientation='horizontal')
                cb1.set_ticks(bounds)
                cb1.set_ticklabels([str(int(ibound)) for ibound in bounds])
                cb1.ax.tick_params(labelsize=5, width=0.25, direction='in')
                cb1.outline.set_visible(False)
                cb1.set_label('[%]', fontsize=5)#, labelpad=-0.6)

                figurename = f'globMap.UNCSRC.ensemble{ensemble_type}_{name}_{drought_severity}_{variable}_{prd}_{figure_type}.'
                figure_directory = genpath(figure_directory_main, scenario, ensemble_type, variable, distribution, str(scale))
                if not os.path.isdir(figure_directory): os.makedirs(figure_directory)
                figurepath = genpath(figure_directory, figurename)

                for suffix in suffixes:
                    if os.path.exists(figurepath+suffix): 
                        print('File exists, will be overwritten.')
                    plt.savefig(figurepath+suffix, dpi=savefig_dpi)
                    print('savefig: {}\n'.format(figurepath+suffix))
                plt.close(4)

            # raw_input("Press key to exit...")
            endTime  = datetime.datetime.now()
            diffTime = endTime - strTime
            print('end @', endTime.strftime("%Y-%m-%d %H:%M:%S"))
            print('took {} min in total.'.format(int(diffTime.seconds/60)))

    print('This process successfully finished!!  d(^o^)b)')


if __name__=='__main__':
    main(*sys.argv)


