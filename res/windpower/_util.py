from res.util import *
from res.weather import windutil, NCSource
from res.weather.sources import loadWeatherSource

##################################################
## Make a turbine model library
TurbineInfo = namedtuple('TurbineInfo','profile meta')
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
        power = np.array([(ws,output) for i,ws,output in tmp.iloc[:,:2].itertuples()])
    
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
TurbineLibrary['Performance'] = [x.profile for x in tmp]

