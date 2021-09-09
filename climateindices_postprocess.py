# To: post-process for climateindice. to create drougnt proxy.
#       1. monthly info:
#           1.1.1 monthly drought intensity
#           1.1.2 monthly accumulated intensity
#           1.2.1 monthly drought flug (for a specific threshold)
#           1.2.2 number of this month in a event
#       2. event info
#           2.1   event accumulated intensity
#           2.2.1 event onset
#           2.2.2 event end
#           2.3   event length
#       3. periods' info
#           3.1   period total accumulated intensity
#           3.2   period total drought months
#           3.3   the number of events in a period (frequency)
#           3.4   period average event lentgh
#           3.5   period event length variance
# By: Yusuke Satoh (NIES)
# Note: This script is for scfrs

import os
import sys
import itertools
import numpy as np
import numpy.ma as ma
from datetime import datetime, date
from netCDF4 import Dataset 
from utiltools import fillinMasked, extractMasked
from os.path import join as genpath


# select for your interest ----------------------------------------------------------------------------------------------------------------------
DUMMY = False
LOUD = False

#variables = ['pr']
#ghms = ['for_pr']

#variables = ['qtot', 'soilmoist']
#variables = ['qtot']
#variables = ['soilmoist']
variables = [sys.argv[1]]

#ghms = ['matsiro', 'cwatm', 'clm45', 'h08', 'jules-w1', 'lpjml', 'pcr-globwb', 'watergap2']
ghms = [sys.argv[2]]

#gcms  = ['hadgem2-es','ipsl-cm5a-lr','gfdl-esm2m','miroc5']
gcms = [sys.argv[3]]

#scenarios  = ['rcp26', 'rcp60', 'rcp85']
scenarios = [sys.argv[4]]

season = sys.argv[5]

#drought_severities = ['mild', 'moderate', 'severe', 'extreme']
#drought_severities = ['severe', 'extreme']
drought_severities = ['severe']
#drought_severities = ['mild']

climatology = '30yrs'; periods = ['base_period', 'recent30yrs', 'nearfuture', 'farfuture']
#climatology = '50yrs'; periods = ['base_period', '1st-half-21C', '2nd-half-21C']
nperiod = len(periods)

scales = [1, 3, 6, 12]

#distributions = ['gamma', 'pearson']
distributions = ['gamma']


# Basically, you don't need to edit hear -----------------------------------------------------------------
base = 'base_1861-1960'
version = 'v2'
complevel = 5

dict_soc = {      # hist      rcp
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

dict_threshold = {
    'mild':         0,
    'moderate': -0.99,
    'severe':   -1.49,
    'extreme':  -1.99,
    }

dict_index = {
    'monthly_drought_flug':          ['-'],
    'monthly_severity':              ['-'],
    'monthly_accum_intensity':       ['-'],
    'monthly_number_in_an_event':    ['th'],
    'event_accum_intensity':         ['-'],
    'event_onset':                   ['eventID'],
    'event_end':                     ['eventID'],
    'event_length':                  ['months'],
    'period_total_accum_intensity':  ['-'],
    'period_total_drought_months':   ['months'],
    'period_total_number_of_events': ['times'],
    'period_event_length_average':   ['months'],
    'period_event_lentgh_variance':  ['months'],
    }

input_directory_main = f'/data/rg001/sgec0017/data/isimip2b.droughtpropa/climate_indices_out/{base}/{version}'
output_directory_main = f'/data/rg001/sgec0017/data/isimip2b.droughtpropa/climate_indices_postprocessed/{base}/{version}/climatology_{climatology}'
lndmskPath = '/data/rg001/sgec0017/data/isimip2b/in/landseamask/ISIMIP2b_landseamask_generic.nc4'
grlmskPath = '/data/rg001/sgec0017/data/mapmask/GAUL/flt/gaul2014_05deg.flt'      # GreenLand is 98
grdaraPath = '/data/rg001/sgec0017/data/mapmask/grd_ara.hlf'

nghm, ngcm = len(ghms), len(gcms)
nmembers = nghm*ngcm
ny, nx, ntime = 360, 720, 2868

grl_mask = ma.masked_equal(np.fromfile(grlmskPath, 'float32').reshape(360,720),98).mask
lnd_mask = Dataset(lndmskPath)['LSM'][:][0].mask
lnd_mask = ma.mask_or(lnd_mask, grl_mask)
nlndgrid = 360*720 - lnd_mask.sum()


# ----------------------------------------------------------------------------------------------------------------------
def read_netcdf(variable, scale, distribution, ghm, gcm, scenario):

    if not DUMMY:  # src.shape (ny, nx, ntime)  ntime = 2868 = 12*239
        if variable == 'pr':
            filename = f'{gcm}_hist{scenario}_pr_monthly_1861_2099_spi_{distribution}_{scale:02}.nc'
        elif variable == 'qtot' or variable == 'soilmoist':
            filename = f'{ghm}_{gcm}_hist{scenario}_{dict_soc[ghm][0]}_{dict_soc[ghm][1]}_co2_{variable}_monthly_1861_2099_spi_{distribution}_{scale:02}.nc'
        else:
            raise ValueError('Error. Check variable name.')
        srcpath = genpath(input_directory_main, variable, filename)
        if not os.path.isfile(srcpath):
            raise FileNotFoundError(f'Error!! {srcpath} is not exist... Check!!')
        else:
            print(f'loading... {srcpath}')
            srcs = Dataset(srcpath)[f'spi_{distribution}_{scale:02}'][:]  # (ny, nx, ntime)
            srcs = extractMasked(np.transpose(srcs, axes=(2,0,1)), lnd_mask)  # (ntime, nland)  ntime = 12 x nyear
            return srcs  # (nstep, nland)
    else:
        print(f'generate DUMMY src...  ({nlndgrid}, {ntime})')
        return np.random.rand(ntime, nlndgrid)


# ----------------------------------------------------------------------------------------------------------------------
def extract_event_length(event_id, event_length):

    if event_id < 0:
        return 0

    event_length = ma.masked_equal(event_length, 0).compressed()
    sample_size = len(event_length.tolist())
    if not event_id > sample_size:
        if not len(event_length.tolist()) == 0:
            value = event_length[event_id-1]
        else:
            value = 0
        return value
    else:
        print('WARNING!!! Something with wrong??   event_id > sample_size')
        return 0


# ----------------------------------------------------------------------------------------------------------------------
def write_netcdf(name, src, scenario, variable, distribution, scale, drought_severity, gcm, ghm, season):

    print(f'writing... {name} {scenario} {variable} {distribution} {scale} {drought_severity} {gcm} {ghm} {season}')

    outputunit = dict_index[name][0]
    mask = Dataset(lndmskPath)['LSM'][:][0].mask

    # convert from land-only to global
    src = fillinMasked(src, lnd_mask)
    lats = np.arange(89.75, -90, -0.5)
    lons = np.arange(-179.75, 180, 0.5)
    if not 'period' in name:
        Times = [(date(iyear,imon,1)-date(1661,1,1)).days for iyear in range(1861, 2099+1) for imon in range(1,13)]
    else:
        Times = [(date(dict_period[period][0],1,1)-date(1661,1,1)).days for period in periods]
    nT = len(Times)
    print(f'lats:  {len(lats)}\nlons:  {len(lons)}\nnT:   {len(Times)}')

    # open a netcdf and write-in
    if variable == 'pr':
        filename = f'{gcm}_hist{scenario}_pr_monthly_1861_2099_spi_{distribution}_{scale:02}_{drought_severity}_{name}.nc'
    else:
        soc_hist, soc_future = dict_soc[ghm]
        filename = f'{ghm}_{gcm}_hist{scenario}_{soc_hist}_{soc_future}_co2_{variable}_monthly_1861_2099_spi_{distribution}_{scale:02}_{drought_severity}_{name}.nc'
    if season == 'ANN':
        output_directory = genpath(output_directory_main, variable)
    else:
        output_directory = genpath(output_directory_main, variable, season)
    if not os.path.isdir(output_directory): os.makedirs(output_directory)
    outputpath = genpath(output_directory, filename)

    rootgrp = Dataset(outputpath, 'w', format='NETCDF4')

    rootgrp.description = 'ISIMIP2b drought propagetion analysis'
    import time
    rootgrp.history     = 'Created ' + time.ctime(time.time())
    if ghm in dict_soc.keys():
        rootgrp.source      = f'ISI-MIP2b : {ghm}_{gcm}_{scenario}_{dict_soc[ghm]}'
    else:
        rootgrp.source      = f'ISI-MIP2b : {gcm}_{scenario}'
    rootgrp.title       = name
    rootgrp.institution = 'NIES'
    rootgrp.contact     = 'satoh.yusuke@nies.go.jp'
    rootgrp.version     = version

    time = rootgrp.createDimension('time', nT)
    lon  = rootgrp.createDimension('lon', 720)
    lat  = rootgrp.createDimension('lat', 360)

    times                    = rootgrp.createVariable('time','f8',('time',), zlib=True, complevel=complevel)
    times.units              = 'days since 1661-01-01 00:00:00'

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

    srcs                     = rootgrp.createVariable(name, 'f4', ('time','lat','lon'), 
                                                      zlib=True, complevel=complevel,
                                                      fill_value=1.e+20,
                                                      chunksizes=(1, 360, 720)
                                                      )
    srcs.units               = outputunit
    srcs.missing_value       = np.float32(1.e+20)

    print(f'latitudes[:] {latitudes[:].shape}')
    print(f'lats         {lats.shape}')
    print(f'times        {times.shape}')
    times[:]      = Times
    latitudes[:]  = lats
    longitudes[:] = lons
    srcs[:]       = src

    print('\n===Attribute===')
    #for name in rootgrp.ncattrs() :
    #    print name, '=', getattr(rootgrp,name)
    print('rootgrp.description      = {}'.format(rootgrp.description))
    print('rootgrp.history          = {}'.format(rootgrp.history))
    print('rootgrp.source           = {}'.format(rootgrp.source))
    print('longitudes.long_name     = {}'.format(longitudes.long_name))
    print('longitudes.units         = {}'.format(longitudes.units))
    print('longitudes.standard_name = {}'.format(longitudes.standard_name))
    print('longitudes.axis          = {}'.format(longitudes.axis))
    print('latitudes.long_name      = {}'.format(latitudes.long_name))
    print('latitudes.units          = {}'.format(latitudes.units))
    print('latitudes.standard_name  = {}'.format(latitudes.standard_name))
    print('latitudes.axis           = {}'.format(latitudes.axis))
    print('times.units              = {}'.format(times.units))
    print('srcs.units               = {}'.format(srcs.units))

    rootgrp.close()
    print(f'\nFinished writting   : {outputpath} {src.shape} {src.min()}-{src.max()}\n')


# ----------------------------------------------------------------------------------------------------------------------
def main(*args):

    strTime = datetime.now()
    print('\nSTART postprocess_climateincides.py\n@{}'.format(strTime.strftime("%Y-%m-%d %H:%M:%S")))

    for scenario, variable, distribution, scale, drought_severity, gcm, ghm in itertools.product(scenarios, variables, 
                                                                                                 distributions, scales,
                                                                                                 drought_severities,
                                                                                                 gcms, ghms):
        print(f'Process: {scenario}, {variable}, {distribution}, {scale}, {drought_severity}, {gcm}, {ghm}')

        spi = read_netcdf(variable, scale, distribution, ghm, gcm, scenario)  # (ntime, nland)
        print(f'read spi data: {spi.shape}')
        shp = spi.shape
        nstep, nland = shp
        nyear = nstep / 12.

        # monthly info
        monthly_drought_flug = np.where(spi < dict_threshold[drought_severity], 1, 0)        # (ntime, nland)
        monthly_severity = np.where(spi < dict_threshold[drought_severity], np.abs(spi), 0)  # (ntime, nland)

        # -- empty fields --- #
        monthly_accum_intensity    = np.zeros(shp, 'float32')                             # (ntime, nland)
        monthly_number_in_an_event = np.zeros(shp, 'float32')                             # (ntime, nland)
        # period info
        period_total_accum_intensity  = np.zeros((nperiod, nland), 'float32')
        period_total_drought_months   = np.zeros((nperiod, nland), 'float32')
        period_total_number_of_events = np.zeros((nperiod, nland), 'float32')
        if season == 'ANN':
            # event info
            event_accum_intensity        = np.zeros(shp, 'float32')
            event_onset                  = np.zeros(shp, 'float32')
            event_end                    = np.zeros(shp, 'float32')
            event_length                 = np.zeros(shp, 'float32')
            period_event_length_average  = np.zeros((nperiod, nland), 'float32')
            period_event_lentgh_variance = np.zeros((nperiod, nland), 'float32')

        # main process --------------
        if season == 'ANN':
            for iland in range(nland):

                event_onset_counter = 0
                event_end_counter   = 0
                for istep in range(nstep):

                    if monthly_drought_flug[istep, iland] == 1:  # if drought

                        if istep == 0:  # first stap
                            monthly_number_in_an_event[0, iland] = monthly_drought_flug[istep, iland]
                            monthly_accum_intensity[0, iland] = monthly_severity[istep, iland]
                        else:  # accumulate values...
                            monthly_number_in_an_event[istep, iland] = monthly_number_in_an_event[istep-1, iland] + monthly_drought_flug[istep, iland]
                            monthly_accum_intensity[istep, iland] = monthly_accum_intensity[istep-1, iland] + monthly_severity[istep, iland]
                            # Always the last step of a event (i.e "event_end") has this accumulated value.
                            event_accum_intensity[istep-1, iland] = 0  # reset
                            event_accum_intensity[istep, iland] = monthly_accum_intensity[istep, iland].copy()
                            # Always the last step of a event (i.e "event_end") has this accumulated value
                            event_length[istep-1, iland] = 0  # reset
                            event_length[istep, iland] = monthly_number_in_an_event[istep, iland].copy()

                    # search onset and end, and give them event IDs
                    #    event:  ...---- -----    --  -------  -----       ------  ---
                    #    onset:  ...00000300000000400050000000060000000000070000000800...
                    #    end  :  ...00020000030000040000000050000006000000000000700000...
                    if istep == 0 and monthly_drought_flug[istep, iland] == 1:
                        event_onset_counter += 1
                        event_onset[istep, iland] = event_onset_counter
                    elif monthly_drought_flug[istep-1, iland] == 0 and monthly_drought_flug[istep, iland] == 1:
                        event_onset_counter += 1
                        event_onset[istep, iland] = event_onset_counter
                    if not istep == 0:
                        if monthly_drought_flug[istep-1, iland] == 1 and monthly_drought_flug[istep, iland] == 0:
                            event_end_counter += 1
                            event_end[istep-1, iland] = event_end_counter

        # periods' info
        for iland in range(nland):
            for iperiod, period in enumerate(periods):

                year_index_start = years_full_period.index(dict_period[period][0])
                year_index_end   = years_full_period.index(dict_period[period][1])

                if season == 'ANN':
                    index_start = 12 * year_index_start
                    index_end   = 12 * year_index_end + 11
                    if LOUD: print(f'({season}) period {period}: {index_start} ... {index_end}')
                
                    period_total_drought_months[iperiod, iland] = monthly_drought_flug[index_start:index_end+1, iland].sum(axis=0)
                    period_total_accum_intensity[iperiod, iland] = monthly_severity[index_start:index_end+1, iland].sum(axis=0)

                    first_event_id = np.min([ma.masked_equal(event_onset[index_start:index_end+1, iland], 0).min(),
                                             ma.masked_equal(event_end[index_start:index_end+1, iland], 0).min()
                                             ]).astype('int')
                    last_event_id = np.max([event_onset[index_start:index_end+1, iland].max(), 
                                            event_end[index_start:index_end+1, iland].max()
                                            ]).astype('int')

                    period_total_number_of_events[iperiod, iland] = last_event_id - first_event_id + 1
                    period_event_length_average[iperiod, iland] = np.divide(period_total_drought_months[iperiod, iland], period_total_number_of_events[iperiod, iland])

                    if not first_event_id < 0:
                        period_event_lentgh_variance[iperiod, iland] = np.var([extract_event_length(event_id, event_length[:, iland])
                                                                                for event_id in range(first_event_id, last_event_id+1)])
                else:
                    index_start = year_index_start
                    index_end   = year_index_end
                    if LOUD: print(f'({season}) period {period}: {index_start} ... {index_end}')
                    monthly_drought_flug_at = monthly_drought_flug[:, iland].reshape(-1,12)  # (nyear, 12)
                    monthly_severity_at     = monthly_severity[:, iland].reshape(-1,12)      # (nyear, 12)
                    if season == 'JJA':
                        period_total_drought_months[iperiod, iland]  = monthly_drought_flug_at[index_start:index_end+1, 5:8].sum()
                        period_total_accum_intensity[iperiod, iland] = monthly_severity_at[index_start:index_end+1, 5:8].sum()
                    elif season == 'DJF':
                        monthly_drought_flug_at = np.concatenate((monthly_drought_flug_at[:,-1].reshape(nyear,1), monthly_drought_flug_at[:,:2]), axis=1)  # (nyear, 12)
                        monthly_severity_at     = np.concatenate((monthly_severity_at[:,-1].reshape(nyear,1), monthly_severity_at[:,:2]), axis=1)          # (nyear, 12)
                        period_total_drought_months[iperiod, iland]  = monthly_drought_flug_at[index_start:index_end+1, :3].sum()
                        period_total_accum_intensity[iperiod, iland] = monthly_severity_at[index_start:index_end+1, :3].sum()

        # output contents depending on season 
        if season == 'ANN':
            outputs = [
                ('monthly_drought_flug', monthly_drought_flug),
                ('monthly_severity', monthly_severity),
                ('monthly_accum_intensity', monthly_accum_intensity),
                ('monthly_number_in_an_event', monthly_number_in_an_event),
                ('event_accum_intensity', event_accum_intensity),
                ('event_onset', event_onset),
                ('event_end', event_end),
                ('event_length', event_length),
                ('period_total_accum_intensity', period_total_accum_intensity),
                ('period_total_drought_months', period_total_drought_months),
                ('period_total_number_of_events', period_total_number_of_events),
                ('period_event_length_average', period_event_length_average),
                ('period_event_lentgh_variance', period_event_lentgh_variance),
                ]
        else:
            outputs = [
                ('period_total_accum_intensity', period_total_accum_intensity),
                ('period_total_drought_months', period_total_drought_months),
                ]
        # conv from land-only to global and write out resuts
        for name, src in outputs:
            write_netcdf(name, src, scenario, variable, distribution, scale, drought_severity, gcm, ghm, season)

    # raw_input("Press key to exit...")
    endTime  = datetime.now()
    diffTime = endTime - strTime
    print('end @', endTime.strftime("%Y-%m-%d %H:%M:%S"))
    print('took {} min in total.'.format(int(diffTime.seconds/60)))
    print('------------ Well DONE!! d(^.^)b ------------')


if __name__=='__main__':
    main(*sys.argv)


