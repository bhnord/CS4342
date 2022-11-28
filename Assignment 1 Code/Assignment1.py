import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def k_euclidian():
    obs = []
    obs.append(numpy.array((0,3,0))) #red
    obs.append(numpy.array((2,0,0))) #red
    obs.append(numpy.array((0,1,3))) #red
    obs.append(numpy.array((0,1,2))) #green
    obs.append(numpy.array((-1,0,1))) #green
    obs.append(numpy.array((1,1,1))) #red

    point = numpy.array((0,0,0))

    for num, i in enumerate(obs):
        print("Obs {}: distance from {} to {} is {}".format(num+1, point, i, numpy.linalg.norm(point-i)))

def appliedQ1():
    df = pd.read_csv("Auto.csv")


    #remove any non-numerical data (? values)
    df[df.columns[:-2]] = df[df.columns[:-2]].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df.reset_index(drop=True) #I AM ASSUMING THAT THESE DO NOT COUNT AS ""


    #---------1b--------------
    #find ranges of each quantitative row
    ranges = [(i, df[i].max()- df[i].min()) for i in df.iloc[:, :-3]]
    print("Range: {}".format(ranges))

    #---------1c--------------
    #std devs of each quantitative row
    std_devs = [(i, df[i].std()) for i in df.iloc[:, :-3]]
    print("Standard Dev: {}".format(std_devs))
    #mean 
    mean = [(i, df[i].mean()) for i in df.iloc[:, :-3]]
    print("Mean: {}".format(mean) )



    #---------1d--------------
    #drop rows 10-85 -- (indexing starts at 0, so range is 9-84 inclusive)
    df2 = df.drop(labels = list(range(9, 85)))

    #find ranges of each quantitative row
    ranges = [(i, df2[i].max()- df2[i].min()) for i in df2.iloc[:, :-3]]
    print("Range: {}".format(ranges))

    #std devs of each quantitative row
    std_devs = [(i, df2[i].std()) for i in df2.iloc[:, :-3]]
    print("Standard Dev: {}".format(std_devs))

    #mean for each quantitative row
    mean = [(i, df2[i].mean()) for i in df2.iloc[:, :-3]]
    print("Mean: {}".format(mean))

    #---------1e+f--------------
    g = sns.PairGrid(df)
    g = g.map(plt.scatter)
    plt.show()

def appliedQ2():
    #---------2--------------
    df = pd.read_csv("Boston.csv", index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df.reset_index(drop=True) 

    #---------2a--------------
    print("Rows: {}, Columns {}".format(len(df.index), len(df.columns)) )

    #---------2bcd--------------
    g = sns.PairGrid(df)
    g = g.map(plt.scatter, s=4)
    plt.show()

    #---------2d--------------
    ranges = [(i, df[i].max(),df[i].min(), df[i].max()- df[i].min()) for i in df]
    print("Range: {}".format(ranges))
    #---------2e--------------
    print("Bounding Charles River: {}".format(len(df.loc[df['chas']==1])))

    print("PtRatio median: {}".format(df['ptratio'].median())) 

    print("Lowest medv: \n{}".format(df.loc[df['medv'] == df['medv'].min()]))
    print("Medv median: {}, medv mean {}".format(df['medv'].median(), df['medv'].mean()))


    print(">7 rooms per dwelling: {}".format(len(df.loc[df['rm'] > 7])))
    print(">8 rooms per dwelling: {}".format(len(df.loc[df['rm'] > 8])))
    print(">8 rooms per dwelling: \n{}".format(df.loc[df['rm'] > 8]))

def math_and_probability():
    for i in range(1, 4):
        d = numpy.random.normal(-3, numpy.sqrt(10), 10**i)
        df = pd.DataFrame(d, columns=['samples'])
        print("{} ---- Mean: {}, Variance: {}".format(10**i, df['samples'].mean(), df["samples"].var()))
        df.plot(kind='hist')
    plt.show()

appliedQ1()
appliedQ2()
