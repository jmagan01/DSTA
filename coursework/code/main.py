'''
March 23. 2019. London, UK. Data Science and Techniques @ Birkbeck College.
Exercise on implementation of PCA for Coursework II.
''' 

# Import the needed libraries and modules
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing
from dfply import *

# Define some constants to be used later in the script
FONTSIZETITLE = 14
FONTSIZE = 11
FIGSIZE = (8, 6)
ALPHA = 0.3
BINS = 15

'''
This function returns a nice-looking scatter plot. It receives 3 args:
(1) a pandas dataframe, (2) x_var name, (3) y_var name and (4) a group variable.
''' 
def scatter_plot(df, x_var, y_var, group_var):
    plt.clf()
    plt.figure(1, figsize = FIGSIZE)
    scatterplot = plt.scatter(df[x_var], df[y_var], c = df[group_var], cmap = plt.cm.Set1)
    plt.xlabel(x_var, fontsize = FONTSIZE)
    plt.ylabel(y_var, fontsize = FONTSIZE)
    return scatterplot

def main():

    # Cosmetic style for plots
    plt.style.use('seaborn-white')

    # Set working directory - change as needed before running the program -
    os.chdir("C:/Users/910589/Desktop/DSTA/coursework")

    # Load a CSV files as pandas dataframes
    sales_df    = pd.read_csv('data/train.csv')
    features_df = pd.read_csv('data/features.csv')
    stores_df   = pd.read_csv('data/stores.csv', dtype={'Type': 'category'}) 


    # Compute the total sales per store
    sales_df = (sales_df >>
      select(X.Store, X.Weekly_Sales) >>
      group_by(X.Store) >>
      summarize(Sales = X.Weekly_Sales.sum()))

    
    # Encode 'Type' of store variable as numeric variable
    stores_df['Type_of_store'] = stores_df['Type'].cat.codes


    # Compute the total Markdown per store
    markdown_df = (features_df >>
      select(X.Store, X.MarkDown5) >>
      group_by(X.Store) >>
      summarize(Markdown = X.MarkDown5.sum()))


    # Merge the stores_df and sales data
    stores_df = pd.merge(stores_df, sales_df, 
      how = 'left', left_on = ['Store'], right_on = ['Store'])


    # Merge the stores_df and markdown data
    stores_df = pd.merge(stores_df, markdown_df, 
      how = 'left', left_on = ['Store'], right_on = ['Store'])


    # Normalise the data
    cols = ['Size', 'Sales', 'Markdown']
    stores_df[cols] = stores_df[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    stores_df = stores_df.loc[:, stores_df.columns != 'Type']


    # Plot a histogram for each selected variable
    plt.clf()
    
    # Set common labels for the 3 subplots
    fig = plt.figure(1, figsize = FIGSIZE)
    plt.suptitle('Distribution of selected dimensions (normalised)', fontsize = FONTSIZETITLE)
    ax = fig.add_subplot(111, frameon = False) # big subplot
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_ylabel('frequency', fontsize = FONTSIZE, labelpad = 25)

    # Create subplots
    kwargs = dict(histtype = 'bar', alpha = ALPHA, bins = BINS, ec = "k")
    i = 1
    for col in cols:
      sub = fig.add_subplot(3, 1, i, frameon = False) # add subplot
      sub.set_title(col, fontsize = FONTSIZE) # subplot's title
      plt.hist(stores_df[col], **kwargs)
      i += 1
    fig.subplots_adjust(hspace = 0.5)
    plt.savefig('histograms.png')
    plt.show()


    # Create and save scatter plots
    scatter_plot(stores_df, 'Sales', 'Size', 'Type_of_store')
    plt.savefig('scatter_sales_vs_size')
    scatter_plot(stores_df, 'Sales', 'Markdown', 'Type_of_store')
    plt.savefig('scatter_sales_vs_total_markdown')

    # To getter a better understanding of interaction of the dimensions, creare a plot in 3D.
    fig = plt.figure(1, figsize = FIGSIZE)
    ax = Axes3D(fig, elev = -150, azim = 110)
    ax.scatter(stores_df['Size'], stores_df['Sales'], stores_df['Markdown'],
               c = stores_df['Type_of_store'], cmap = plt.cm.Set1, edgecolor = 'k', s = 40)
    ax.set_title("Selected dimensions", fontsize = FONTSIZETITLE)
    ax.set_xlabel("Size of store", fontsize = FONTSIZE)
    ax.set_ylabel("Sales", fontsize = FONTSIZE)
    ax.set_zlabel("Markdown (discounts)", fontsize = FONTSIZE)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()


    # Compute first 3 PCA dimensions.
    fig = plt.figure(1, figsize = FIGSIZE)
    ax = Axes3D(fig, elev = -150, azim = 110)
    X_reduced = PCA(n_components = 3).fit_transform(stores_df)

    # Plot the first 3 PCA dimensions.
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
               c = stores_df['Type_of_store'], cmap = plt.cm.Set1, edgecolor = 'k', s = 40)
    ax.set_title("First three PCA dimensions", fontsize = FONTSIZETITLE)
    ax.set_xlabel("1st eigenvector", fontsize = FONTSIZE)
    ax.set_ylabel("2nd eigenvector", fontsize = FONTSIZE)
    ax.set_zlabel("3rd eigenvector", fontsize = FONTSIZE)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()

if __name__ == "__main__":

    main()
