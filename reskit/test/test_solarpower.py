def test_my_sapm_celltemp():
    # (poa_global, wind_speed, temp_air, model='open_rack_cell_glassback'):
    print( "my_sapm_celltemp not tested..." )

def test_spencerSolPos():
    # (times, lat, lon):
    print( "spencerSolPos not tested..." )

def test_myDisc():
    # (ghi, zenith, I0, am, pressure):
    print( "myDisc not tested..." )

def test_myDirint():
    # (ghi, zenith, pressure, use_delta_kt_prime, temp_dew, amRel, I0):
    print( "myDirint not tested..." )

def test_ensureSeries():
    # (var, locs):
    print( "ensureSeries not tested..." )

def test_frankCorrectionFactors():
    # (ghi, dni_extra, times, solarElevation):
    print( "frankCorrectionFactors not tested..." )

def test_locToTilt():
    # (locs, convention="latitude*0.76", **k):
    print( "locToTilt not tested..." )

def test__presim():
    # (locs, source, elev=300, module="WINAICO WSx-240P6", azimuth=180, tilt="ninja", totalSystemCapacity=None, tracking="fixed", modulesPerString=1, inverter=None, stringsPerInverter=1, rackingModel='open_rack_cell_glassback', airmassModel='kastenyoung1989', transpositionModel='perez', cellTempModel="sandia", generationModel="single-diode", inverterModel="sandia", interpolation="bilinear", loss=0.16, trackingGCR=2/7, trackingMaxAngle=60, frankCorrection=False):
    print( "_presim not tested..." )

def test_my_i_from_v():
    # (resistance_shunt, resistance_series, nNsVth, voltage):
    print( "my_i_from_v not tested..." )

def test_my_golden_sect_DataFrame():
    # (params, VL, VH, func):
    print( "my_golden_sect_DataFrame not tested..." )

def test_my_pwr_optfcn():
    # (df, loc):
    print( "my_pwr_optfcn not tested..." )

def test_mysinglediode():
    # (photocurrent, saturation_current, resistance_series):
    print( "mysinglediode not tested..." )

def test_simulation():
    # (singleAxis, tilt, module, azimuth, inverter, moduleCap, modulesPerString, stringsPerInverter, locs, times, dni, ghi, dhi, amRel, solpos, pressure, air_temp, windspeed, dni_extra, sandiaCellTemp, transpositionModel, totalSystemCapacity, sandiaGenerationModel, loss, approximateSingleDiode, goodTimes):
    print( "simulation not tested..." )

def test_simulatePVModule():
    # (locs, source, elev=300, module="WINAICO WSx-240P6", azimuth=180, tilt="ninja", totalSystemCapacity=None, tracking="fixed", interpolation="bilinear", loss=0.16, rackingModel="open_rack_cell_glassback", approximateSingleDiode=True, **kwargs):
    print( "simulatePVModule not tested..." )

def test_simulatePVModuleDistribution():
    # (locs, tilts, source, elev=300, azimuths=180, occurrence=None, rackingModel="roof_mount_cell_glassback", module="LG Electronics LG370Q1C-A5", approximateSingleDiode=True, **kwargs):
    print( "simulatePVModuleDistribution not tested..." )


if __name__ == "__main__":
    test_my_sapm_celltemp()
    test_spencerSolPos()
    test_myDisc()
    test_myDirint()
    test_ensureSeries()
    test_frankCorrectionFactors()
    test_locToTilt()
    test__presim()
    test_my_i_from_v()
    test_my_golden_sect_DataFrame()
    test_my_pwr_optfcn()
    test_mysinglediode()
    test_simulation()
    test_simulatePVModule()
    test_simulatePVModuleDistribution()
    