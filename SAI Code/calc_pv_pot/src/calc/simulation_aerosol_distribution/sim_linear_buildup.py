import numpy as np
import pandas as pd


def linear_buildup(injection, distr_init, injection_index):

    buildup_step = injection / 4                         # 1/4 der max. injection wird für 4 monate hinzugefügt

    distr_new = np.copy(distr_init)                      # copy of initial distr.  

    distr_4month = []

    for month in range(4):
        
        distr_new[injection_index] += buildup_step           # an der injection-region wird build-up menge hinzugefügt
        
        distr_4month.append(np.copy(distr_new))
                                                        
    return distr_4month                                    # Rückgabe liste von distr-array der ersten vier Monate (linear buildup, kein e-folding, kein Austausch)
