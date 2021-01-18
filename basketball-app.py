import streamlit as st
import pandas as pd
from PIL import Image
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

image = Image.open("bbr-logo.png")
st.image(image, use_column_width=False)

st.title('NBA Player Stats Explorer')
st.markdown("""
This app performs simple web-scraping of NBA player stats data!
* **Python Libraries:** base64, pandas, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header('Filters')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2020))))


# loading data from https://www.basketball-reference.com based on query selected by user
@st.cache
def data_loader(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    # raw = df.drop(df[df.Age == "Age"].index)
    raw = df.fillna(0)
    playerstats = raw.drop(["Rk"], axis=1)
    return playerstats


playerstats = data_loader(selected_year)

# team filtering
sorted_unique_teams = sorted(playerstats.Tm.unique())
# st.write(sorted_unique_teams)

# position filtering
unique_positions = ['C', 'PF', 'SF', 'PG', 'SG']
# st.write(unique_positions)

# options for preselection of data
if st.sidebar.checkbox('Select all teams', value=True):
    selected_team = st.sidebar.multiselect("Teams", sorted_unique_teams, sorted_unique_teams)
else:
    selected_team = st.sidebar.multiselect("Teams", sorted_unique_teams)
if st.sidebar.checkbox('Select all Positions', value=True):
    selected_pos = st.sidebar.multiselect("Positions", unique_positions, unique_positions)
else:
    selected_pos = st.sidebar.multiselect("Positions", unique_positions)

# df based on our selection
df_selected = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected.shape[0]) + ' rows and ' + str(df_selected.shape[1]) + ' columns.')
st.dataframe(df_selected)


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download selected data</a>'
    return href


st.markdown(filedownload(df_selected), unsafe_allow_html=True)

if st.button("Intercorrelation Heatmap"):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(fig)
