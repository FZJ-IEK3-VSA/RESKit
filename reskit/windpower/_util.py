from reskit.util.util_ import *
from reskit.weather import windutil, NCSource

##################################################
## Make a turbine model library
TurbineInfo = namedtuple('TurbineInfo','profile meta')
_P = namedtuple('PowerCurve','ws cf')
class PowerCurve(_P):
    """ 
    A wind turbine's power curve represented by a set of (wind-speed,capacty-factor) pairs:
      ws ->  "wind speed" 
      cf ->  "capacity factor" 
    """
    def __str__(s):
        out = ""
        for ws,cf in zip(s.ws, s.cf):
            out += "%6.2f - %4.2f\n"%(ws,cf)
        return out

    def _repr_svg_(s): 
        #return str(s)

        import matplotlib.pyplot as plt
        from io import BytesIO

        plt.figure(figsize=(7,3))
        plt.plot(s.ws,s.cf, color=(0,91/255,130/255), linewidth=3)
        plt.tick_params(labelsize=12)
        plt.xlabel("wind speed [m/s]",fontsize=13)
        plt.ylabel("capacity output",fontsize=13)
        plt.tight_layout()
        plt.grid()
        
        f = BytesIO()
        plt.savefig(f, format="svg", dpi=100)
        plt.close()
        f.seek(0)
        return f.read().decode('ascii')


def lowGenCorrection(capacityfactors, base=0, sharpness=5):
    """Performs capacity factor correction to suppress the generation during low generation times


    Uses the equation:
    .. math::
        \\mathrm{new_cap_fac} = mathrm{original_cap_fac} * ((1-base)*(1-exp(-sharpness*mathrm{original_cap_fac}))+base)
    """


    if isinstance(capacityfactors, PowerCurve):
        _ws = capacityfactors.ws
        capacityfactors = capacityfactors.cf
        asPowerCurve = True
    else:
        asPowerCurve = False

    factors = (1-base)*(1-np.exp(-sharpness*capacityfactors))+base # dampens lower wind speeds
    capacityfactors = factors*capacityfactors
    
    if asPowerCurve:
        return PowerCurve(_ws, capacityfactors)
    else:
        return capacityfactors


rangeRE = re.compile("([0-9.]{1,})-([0-9.]{1,})")
def parse_turbine(path):
    meta = OrderedDict()
    with open(path) as fin:
        # Meta extraction mode
        while True:
            line = fin.readline()[:-1]

            if line == "" or line[0]=="#": continue # skip blank lines and comment lines
            if 'power curve' in line.lower(): break

            sLine = line.split(',')
            if sLine[0].lower()=="hubheight" or sLine[0].lower()=="hub_height":
                heights = []
                for h in sLine[1:]:
                    h = h.replace("\"","")
                    h = h.strip()
                    h = h.replace(" ","")

                    try:
                        h = float(h)
                        heights.append(h)
                    except:
                        try:
                            a,b = rangeRE.search(h).groups()
                            a = int(a)
                            b = int(b)

                            for hh in range(a,b+1):
                                heights.append(hh)
                        except:
                            raise RuntimeError("Could not understand heights")

                meta["Hub_Height"] = np.array(heights)
            else:
                try:
                    meta[sLine[0].title()] = float(sLine[1])
                except:
                    meta[sLine[0].title()] = sLine[1]
        
        # Extract power profile
        tmp = pd.read_csv(fin)
        tmp = np.array([(ws,output) for i,ws,output in tmp.iloc[:,:2].itertuples()])
        power = PowerCurve( tmp[:,0], tmp[:,1]/meta["Capacity"] )
    return TurbineInfo(power, meta)     

turbineFiles = glob(join(dirname(__file__),"..","..","data","turbines","*.csv"))

tmp = []
for f in turbineFiles:
    try:
        tmp.append(parse_turbine(f))
    except:
        print("failed to parse:", f)

TurbineLibrary = pd.DataFrame([i.meta for i in tmp])
TurbineLibrary.set_index('Model', inplace=True)
TurbineLibrary['PowerCurve'] = [x.profile for x in tmp]

#######################################################
#### Create a synthetic turbine power curve
synthTurbData = pd.read_csv(join(dirname(__file__),"..","..","data","synthetic_turbine_params.csv"), header=1)

def SyntheticPowerCurve( specificCapacity=None, capacity=None, rotordiam=None, cutout=25 ):
    """The synthetic power curve generator creates a wind turbine power curve 
    based off observed relationships between turbine specific power and known
    power curves
    """
    if cutout is None: cutout=25
    if specificCapacity is None:
        specificCapacity = capacity*1000/(np.pi*rotordiam**2/4)
        
    specificCapacity = int(specificCapacity)
    # Create ws
    ws = [0,]
    ws.extend( np.exp(synthTurbData.const + synthTurbData.scale*np.log(specificCapacity)) )
    ws.extend( np.linspace(ws[-1], cutout, 20)[1:])
    ws = np.array(ws)
    
    # create capacity factor output
    cf = [0,]
    cf.extend( synthTurbData.perc_capacity/100 )
    cf.extend([1]*19)
    cf = np.array(cf)
    
    # Done!
    return PowerCurve(ws, cf)


def specificPower(capacity, rotordiam, **k):
    """Computes specific power from capacity and rotor diameter"""
    return capacity*1000/rotordiam**2/np.pi*4