import streamlit as st
import pandas as pd
import hvplot.pandas
import holoviews as hv
from holoviews import dim, opts
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import psutil

st.set_page_config(page_title="Analysing Shill Bid dataset", page_icon=":tada:", layout="wide")

process = psutil.Process()
memory_usage = process.memory_info().rss  # Memory usage in bytes
memory_usage_mb = memory_usage / (1024 * 1024)  # Convert to megabytes

st.write("This app uses: ", round(memory_usage_mb, 2), "MB of RAM")

# st.markdown('''
# <style>
# .stApp [data-testid="stToolbar"]{
#     display:none;
# }
# </style>
# ''', unsafe_allow_html=True)


# Load data from CSV file
data = pd.read_csv('data.csv')
features = data.drop(['Class', 'Bidder_ID'], axis=1)
####


st.title("Shill Bid dataset analysis :tada:") 
col1, col2 = st.columns(2) 

with col1:
    # Create a slider to select the range of records
    range_slider = st.slider('Select how many records you want to work with shill-bid.csv dataset:', 0, len(data),
    (1, 1000))
    # Get the start and end index from the slider value
    start_index, end_index = range_slider
    # Check if the start and end indices are the same This will avoid errors
    if start_index == end_index:
    # Subtract 100 from either start_index or end_index, taking care not to go out
    # of range

        if start_index > 100: 
            start_index -= 100 
        else: 
            end_index += 100

    # Make sure the indices stay within the range of the DataFrame
    start_index = max(start_index, 0) 
    end_index = min(end_index, len(data))
    # new filtered data
    data = data.iloc[start_index:end_index] 

    downsample = st.checkbox("Downsample Data [for balancing 'Class' feature]")

    if downsample: 

        class_counts = data['Class'].value_counts()
        min_class_count = class_counts.min()
        
        data_downsampled = data.groupby('Class').apply(lambda x: x.sample(min_class_count))
        data_downsampled.reset_index(drop=True, inplace=True)
        data=data_downsampled


with col2: 
        st.write("""At the left side
you can adjust how many rows will be used for the analysis of the shill bid
dataset. This selection will adjust to all the charts and tables that are on the website. 
Also check my [Shill Bid Report](https://drive.google.com/file/d/1Fu4NdN-lJJqCi_VWaSJjWGxcJr292jB9/view)
""")
        st.write("**Currrent data shape**:", data.shape)


tab1, tab2, tab3, tab33, tab4, dclust, tab5, tab6 = st.tabs(["⭐ Introduction", 
                            "🗃 Dataset", 
                            "📊 Histograms", 
                            "📊 Histogram Analyser", 
                            "🇲🇬 Clusters ", 
                            "🇲🇬 3d Cluster", 
                            "🗄 Filtering Shill bids",
                            "🔫 Detecting Shill Bids"])

tab1.write("""## Introduction
Due to the growing popularity of online shopping, bidding has become a popular
technique to assess the true market value of a product. Users can bid on items
on eBay, one of the most famous online retailers, to determine the true value of
their purchases. Nevertheless, some offers were intended to artificially inflate
the price of the item. Bidding, which is often intended to raise the price for
other bidders, is when a bidder bids for something without wanting to win it.

## Goals

The main reason for this web app is the detection of the Shill bids.  
This web app will provide a detailed description and give deep insights into how
Shill bids are spread across the dataset.  

In this panel application, you can explore:

- **Dataset description tab**
    - which explains basic statistics about the Shill bid dataset to give a better understanding of what type of data we are dealing with.
- **Histograms tab**
    - Histograms were plotted to give insights about how the data was spread and give us insights into how the distribution looked like.
- **Histogram Analyser tab**
    Here, we can observe all the histograms and interactively compare them. We can
    see that the majority of values for all the histograms are in the range of 0-0.1
    and 0.9-1. At the bottom of the chart, we can see how dense the variables we are
    working with are.
- **Correlation Matrix**
    - This chart is helping to show correlation across all the dataset feature variables and identify highest correlation with target class variable
and we can see that these features can be the most helpful in identifying shill bids:
        - The Successive-Outbidding (0.9), 
        - Bidding-Ratio (0.57) 
        - Winning-Ratio (0.39)
    - They indicate that these features are the most useful in making our prediction of the Shill bids.
- **Clustering tab**
    - This type of charting is confirming that when Successive Outbidding is equal to or higher than 0.5 is where 80% of all fraudulent bids are located in.Exploring
     clustering charts will give us insight that they are located in a specific way.
- **3d Cluster tab**
    - This type of chart was chosen because we had the 3 most correlated features with the Class. We can see how
    clusters are formed including in the same time 3 of them.
- **Filtering Shill Bids**
    - This interactive tab is presenting with what kind of problem we are dealing here with. And how traditional filtering methods are working and
    that using maximum correlated features with class variable can help us to select up to 85% shill bids.
- **Detecting Shill Bids with Decision Tree**
    - In this interactive tab we can see how Random Forest model is way superior that standard filtering methods. 
    It shows how we can detect up to 99% shill bids. We also can understand how depth hyperparameter
    will change the final accuracy score. 
"""
)
#tab1.line_chart(data)

tab2.write("## Dataset")
tab2.write("Here is how the dataset that we are working with looks like: (it will update if you will move the slider)")
tab2.write(data.head(10))
tab2.write("The shape of this dataset is")
tab2.write(data.shape)
tab2.write("""
- Record ID: Unique identifier of a record in the dataset.
- Auction ID: Unique identifier of an auction.
- Bidder ID: Unique identifier of a bidder.
- Bidder Tendency: A shill bidder participates exclusively in auctions of a few sellers
rather than a diversified lot. This is a collusive act involving the fraudulent seller and
an accomplice.
- Bidding Ratio: A shill bidder participates more frequently to raise the auction price and
attract higher bids from legitimate participants.
- Successive Outbidding: A shill bidder successively outbids himself even though he is
the current winner to increase the price gradually with small consecutive increments.
- Last Bidding: A shill bidder becomes inactive at the last stage of the auction (more than
90 per cent of the auction duration) to avoid winning the auction.
- Auction Bids: Auctions with SB activities tend to have a much higher number of bids
than the average of bids in concurrent auctions.
- Auction Starting Price: a shill bidder usually offers a small starting price to attract
legitimaie bidders into the auction.
- Early Bidding: A shill bidder tends to bid pretty early in the auction (less than 25 per
cent of the auction duration) to get the attention of auction users.
- Winning Ratio: A shill bidder competes in many auctions but hardly wins any auctions.
- Auction Duration: How long an auction lasted.
- Class: 0 for normal behaviour bidding; 1 for otherwise.""")

tab2.write("## Dataset Statistics")
tab2.write(data.describe())
tab2.write("""
By looking
at the min and max values for each feature we can understand that the dataset was already
scaled because values are ranging from 0 to 1. Only the Auction-Duration was left as it is
ranging from 0 to 10 days. We can see that the mean values indicate that most of the variables
have relatively low values (e.g. Bidder-Tendency, Bidding-Ratio and Successive-Outbidding
are all around 0.14, 0.13 and 0.1 respectively) suggesting that there is no normal distribution
of that data points. The class has the lowest mean value at 0.11 indicating that there is a significant amount of
normal bids compared to Shill bids. For the Auction-Duration variable, the mean value is 4.62.
""")

tab3.write(""" # Histograms """)
import plotly.express as px
import pandas as pd

tab3.write("""
- The histogram of Bidder tendency is shifted to the left, values are from 0 to 1 with the
peak of the value around 4000 when it is 0. The majority of samples are in the range of
0 - 0.3
7- Bidding Ratio shows a peak of values around 3700, suggesting that most bidders tend
to participate in auctions less frequently in order to raise the price.
- Majority of users have Successive Outbidding equal to 0, only a small percentage is
equal to 0.5 or 1.
- Last Bidding and Early bidding have similar characteristics with the majority of samples
with values 0 and 1.
- Significant amount of all auctions have Winning Ratio equal to 0 we can interpret it
that most of the users do not win the auction.
    """ )
with tab33:

    import plotly.figure_factory as ff

    # Specify the column names you want to include in the distplot
    column_names = [
        'Record_ID', 'Auction_ID', 'Bidder_Tendency',
        'Bidding_Ratio', 'Successive_Outbidding', 'Last_Bidding',
        'Auction_Bids', 'Starting_Price_Average', 'Early_Bidding',
        'Winning_Ratio'
    ]

    # Set the bin size
    bin_size = 0.1
    st.write('''
    ## Histogram analyser
    

    Here, we can observe all the histograms and interactively compare them. We can
    see that the majority of values for all the histograms are in the range of 0-0.1
    and 0.9-1. At the bottom of the chart, we can see how dense the variables we are
    working with are. For example, the "successive outbidding" feature has only
    three values: 0, 0.5, and 1. You can interact with this chart and select the
    ranges you are interested in. Click on the legend to turn specific features on
    or off in the histograms.
    ''')

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
        [data[column] for column in column_names[2:]],
        column_names[2:],
        bin_size=[bin_size] * len(column_names[2:]),
    )

    
    fig.update_layout(title_text="title", margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=700)

    # Display the plot
    st.plotly_chart(fig, use_container_width=True, use_container_height=True  )

with tab3:
    col1, col2, col3 = st.columns(3)

    # Loop through each column and add a histogram to the tab
    for i, col in enumerate(data.columns):
        fig = px.histogram(data, x=col)
        fig.update_layout(title_text=f"Histogram of {col}")
        
        if i % 3 == 0:
            col1.plotly_chart(fig, use_container_width=True)
        elif i % 3 == 1:
            col2.plotly_chart(fig, use_container_width=True)
        else:
            col3.plotly_chart(fig, use_container_width=True)

with tab4:
    st.write("## 2d Clusters")
    st.write("Red dots are Shill bids that we are looking for. We can see that when the Succesive Outbidding is equal or higher to 0.5 we have cluster of shill bids.  ")

    col1, col2 = st.columns(2)

    fig = px.scatter(
    data,
    x="Successive_Outbidding",
    y="Bidder_Tendency",
    color="Class",
    color_continuous_scale="peach",
)
    fig2 = px.scatter(
    data,
    x="Successive_Outbidding",
    y="Winning_Ratio",
    color="Class",
    color_continuous_scale="peach",
)


    col1.plotly_chart(fig, theme="streamlit", use_container_width=True)
    col2.plotly_chart(fig2, theme="streamlit", use_container_width=True)

with dclust:
    st.write("## 3d Cluster")
    st.write('''On this Chart we can see 3 the most correlated features with
    class variable.  The red most intuitive color indicates a fraudulent bid. Thanks to this
    chart we can see where the fraudulent bids are located in three dimensional
    space.  ''')

    st.info("💡 You can interact with the chart to adjust the best angle for you.")

    import plotly.graph_objs as go

    x = data['Successive_Outbidding']
    y = data['Winning_Ratio']
    z = data['Bidder_Tendency']
    color = data['Class']  # Values used for coloring

    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=5,
            color=color,  # Set color to 'Class' column
            #Choose colorscheme here https://plotly.com/python/builtin-colorscales/
            colorscale="Picnic",
            opacity=0.3,
            showscale=True  # Display color scale in legend
        ),
        name="Data Points"
    )

    data1 = [trace1]
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(title="Legend"),
        scene=dict(
            xaxis_title="Successive_Outbidding",  # Set x-axis label to column name
            yaxis_title="Winning_Ratio",  # Set y-axis label to column name
            zaxis_title="Bidder_Tendency"   # Set z-axis label to column name
        ),
    )

    fig = go.Figure(data=data1, layout=layout)
    st.write(fig)

# Define the slider for filtering

# Create the panel object
with tab5:

    def filter_data(successive_outbidding_value, winning_ratio_value, bidding_ratio_value):
        filtered_data = data.loc[data['Successive_Outbidding'] >= successive_outbidding_value]
        filtered_data = filtered_data.loc[filtered_data['Winning_Ratio'] >= winning_ratio_value]
        filtered_data = filtered_data.loc[filtered_data['Bidding_Ratio'] >= bidding_ratio_value]
        class_counts = filtered_data['Class'].value_counts()
        fraudulent_count = class_counts.get(1, 0)
        total_count = class_counts.sum()
        fraudulent_pct = 100 * fraudulent_count / total_count
        return fraudulent_pct, class_counts

    def plot_pie_chart(class_0_sum, class_1_sum):
        labels = ['Normal Bids', 'Shill Bids', 'Undetected Shill Bids']
        undetected=data['Class'].sum()-class_1_sum
        sizes = [class_0_sum, class_1_sum, undetected]
        explode = (0, 0.2, 0.3)
        colors = ['#66b3ff', '#ff9999', '#ff1119',]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')

        # create pie chart plot
        return fig1

    # Define the function to update the text widget when the slider is changed
    def update_fraudulent_pct():
        col1, col2 = st.columns(2)
        # get the memory usage
        # displaying the memory
        winning_ratio_value = col1.slider(
            'Winning Ratio ', 
            min_value=float(0),
            max_value=float(1),
            step=0.1)
        successive_outbidding_value = col1.slider(
            'Successive Outbidding ', 
            min_value=float(0),
            max_value=float(1),
            step=0.1)
        bidding_ratio_value = col1.slider(
            'Bidding Ratio ', 
            min_value=float(0),
            max_value=float(1),
            step=0.1)
        fraudulent_pct, class_counts = filter_data(successive_outbidding_value, winning_ratio_value, bidding_ratio_value)
        class_0_sum = class_counts.get(0, 0)
        class_1_sum = class_counts.get(1, 0)
        nr_records=data['Class'].shape
        fraud_sum=data['Class'].sum()
        col1.write(f'Records selected: {nr_records}, with {fraud_sum} fraudulend bids')
        col1.write(f'**Filters resulted in selecting:**')
        col1.write(f'Fraudulent bids: {class_1_sum}')
        col1.write(f'Normal bids: {class_0_sum}')
        col1.write(f'Undetected: {fraud_sum-class_1_sum}')
        col2.pyplot(plot_pie_chart(class_0_sum, class_1_sum))
        col2.info('💡 If the chart do not appear click 3d Cluster tab and then "Filtering shill bids" tab again. ')


    st.write('''
## Filtering Shill bids by 3 most correlated features with feature Class

Moving sliders will show how many Shill bids and normal bids will be selected
when the value of the features will change. The goal is to select as many Shill
bids and as few normal bids as possible.  This visualisation is showing with
what kind of challenge we are dealing with in this dataset and that filtering by
the features that are the most correlated with feature class are the most useful
in detecting Shill bids. 
    ''')
    st.markdown('Use the slider to filter bids by [ value >= slider value ]:')
    update_fraudulent_pct()


with tab6:

    # Machine Leraning Models
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import scikitplot as skplt


    col1, col2 = st.columns(2)


    
    col1.write("""
    ## Detecting Shill Bids
    We can see that training a Random Tree classification model on
    the Shill bid dataset is way superior to using traditional approaches
    of using simple filtering methods. 
    Adjust the hyperparameters to re-run the decision tree classifier. The
    training accuracy score will adjust accordingly:""")
    col1.write("---")
    
    X = data.drop(['Class', 'Bidder_ID'], axis=1)
    y = data['Class']



    classifier_name = col1.selectbox(
        'Select classifier',
        ('KNN', 'SVM', 'Random Forest')
    )
    shuffle_state = col1.checkbox("Shuffle Data")

    def add_parameter_ui(clf_name):

      #  memory_usage = process.memory_info().rss 
        params = dict()
        if clf_name == 'SVM':
            C = col1.slider('C', 0.01, 10.0)
            params['C'] = C
        elif clf_name == 'KNN':
            K = col1.slider('K', 1, 10)
            params['K'] = K
        else:
            max_depth = col1.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            n_estimators = col1.slider('n_estimators', 1, 10)
            params['n_estimators'] = n_estimators
        return params

    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'])
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        else:
            clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], random_state=1234)
        return clf

    clf = get_classifier(classifier_name, params)

    #### CLASSIFICATION ####
    test_size=col1.slider("Test size [%]", 0.01,0.99, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234, shuffle=shuffle_state)


    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred),2 )
    recall = round(recall_score(y_test, y_pred), 2)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_labels = ['Normal Bids', 'Fraudulent Bids']

    # Plot the confusion matrix
    # Color refference:
    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    fig, ax = plt.subplots()
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax=ax, cmap="Blues" )

    col2.write('0 - normal, 1-fraud')
    col2.pyplot(fig)

    col2.write("---")
    col2.write('**Evaluation metrics:** ')

    col2.write( f"""
        Accuracy: {acc} \n
        Precision: {precision} \n
        Recall: {recall} \n
        """) 

    col2.write("---")

    # Confusion matrix explanation: 
    true_positive = cm[1][1]  # Correctly predicted shill bids
    false_negative = cm[1][0]  # Incorrectly predicted normal bids as shill bids
    false_positive = cm[0][1]  # Incorrectly predicted shill bids as normal bids
    true_negative = cm[0][0]  # Correctly predicted normal bids

    col1.write("**Confusion Matrix Results:**")
    col1.write(f"- Correctly predicted shill bids: {true_positive}")
    col1.write(f"- Correctly predicted normal bids: {true_negative}")
    col1.write(f"- False positives (Normal bids predicted as shill bids): {false_positive}")
    col1.write(f"- False negatives (Shill bids predicted as normal bids): {false_negative}")
    col1.info(""" 💡 We can see that Random Forest is performing best from all the models. We can be fooled at first if we 
        will use a not balanced dataset that all of the models are performing with high accuracy. However after balancing the dataset 
        we can see real accuracy results. """)



# enable_scroll = """
# <style>
# .main {
#     overflow: auto;
# }
# </style>
# """


# st.markdown(enable_scroll, unsafe_allow_html=True)    

## Footer
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://webtool.page" target="_blank">Marcin Mrugacz</a></p>
</div>
"""

st.markdown(footer,unsafe_allow_html=True)
