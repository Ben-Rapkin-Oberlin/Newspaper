import pandas as pd


df=pd.read_csv("/usr/users/quota/students/2021/brapkin/Newspaper/master_mortality_data_1900_1936_ucb.csv")
df=df[["state_name","year","total_mortality","small_pox"]]
print(df.shape)
a=["Maine","Connecticut","Indiana","Massachusetts","Michigan","New Hampshire","New Jersey","Rhode Island", "New York", ]
df=df[df["state_name"].isin(a)]
print(df.shape)
print(df.columns)

df=df.groupby('year')["small_pox"].sum()

dfstate=pd.read_csv("/usr/users/quota/students/2021/brapkin/Newspaper/State Pops - Sheet1.csv")

#dfstate=dfstate["Total_Ratio"]

print(dfstate.shape)
df['ratio']=dfstate["Total_Ratio"].values



df.to_csv("test.csv")


