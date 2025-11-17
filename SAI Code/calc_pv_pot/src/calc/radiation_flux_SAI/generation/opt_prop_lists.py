
def opt_prop_lists(df, head_index):

    '''
    Funktion nimmt Datensatz und wandlet Spalte in eine Liste um
    Spalten-Kopf ist Name der Liste
    '''    

    list_column_names = df.head() 
    opt_prop = list_column_names[head_index]

    opt_prop = []
    opt_prop.append(df.column(opt_prop))

    return opt_prop