# -*- coding: utf-8 -*-
# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of showing geographic data."""

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
#import streamlit_nested_layout
import streamlit as st
from matplotlib import pyplot as plt
import plotly.express as px

# Helpers
def q05(x): return x.quantile(0.05, interpolation='linear') # 5th Percentile
def q10(x): return x.quantile(0.10, interpolation='linear') # 10th Percentile
def q20(x): return x.quantile(0.20, interpolation='linear') # 20th Percentile
def q32(x): return x.quantile(0.32, interpolation='linear') # -1 sigma
def q50(x): return x.quantile(0.50, interpolation='linear') # median
def q68(x): return x.quantile(0.68, interpolation='linear') # +1 sigma
def q80(x): return x.quantile(0.80, interpolation='linear') # 80th Percentile
def q90(x): return x.quantile(0.90, interpolation='linear') # 90th Percentile
def q95(x): return x.quantile(0.95, interpolation='linear') # 95th Percentile

# Page title
st.set_page_config(layout="wide", page_title="CT MMM Results - Channel one-pager", page_icon=":m::m::m:")

st.warning(":warning: For a PoC, this has been done on my public GitHub profile, but don't worry, this is not real data")

# TODO: read real data from BigQuery or GCS (instead of simulating the data as is done here)
# LOAD DATA ONCE
@st.experimental_singleton
def load_data():
    from datetime import datetime, timedelta
    import pandas as pd
    data = []
    for model_id in ['CT.com', 'B&M']:
        for date in [datetime(2022, 1, 1) + timedelta(days=i) for i in range(0, 365)]:
            for channel in ['YouTube', 'TikTok', 'Bing', 'TV']:
                spend = np.random.gamma(len(channel), 5, 1)[0]
                for optimal_model_id in ['1_111_1', '2_222_2', '3_333_3', '4_444_4', '5_555_5', '6_666_6',
                                         '7_777_7', '8_888_8', '9_999_9']:
                    revenue = np.random.gamma(25, 5, 1)[0]
                    data.append([model_id, optimal_model_id, channel, date, spend, revenue])
    df = pd.DataFrame(data=data, columns=['model_id', 'optimal_model_id', 'channel', 'date', 'spend', 'revenue'])
    df = df.sort_values(by=['model_id', 'optimal_model_id', 'channel', 'date'])

    return df

df_raw = load_data()




with st.sidebar:

    st.header('Parameters')

    MODEL = st.selectbox('Model', df_raw['model_id'].unique())

    CHANNEL = st.selectbox('Channel of focus', df_raw['channel'].unique())

    with st.container():
        col1, col2 = st.columns((1,1))

        with col1:
            DATE_START = st.date_input('Start date', value=df_raw['date'].min())
            DATE_START = np.datetime64(DATE_START)

        with col2:
            DATE_END = st.date_input('End date', value=df_raw['date'].max())
            DATE_END = np.datetime64(DATE_END)

    DATE_AGGREGATION = st.radio('Date aggregation', ('Daily', 'Weekly', 'Monthly', 'Quarterly'), 0)

    MODELS_TO_EXCLUDE = []
    if st.checkbox('Exclude pareto models'):
        MODELS_TO_EXCLUDE = st.multiselect('Exclude pareto models', df_raw['optimal_model_id'].unique())

    st.info(':information_source: the sidebar can be minimised')


# Given options in sidebar, filter and aggregate the data accordingly
df_filtered = df_raw.copy(deep=True)
#  1. Keep selected model
df_filtered = df_filtered[df_filtered['model_id']==MODEL]
#  2. Keep selected channel
df_filtered = df_filtered[df_filtered['channel']==CHANNEL]
#  3. Keep selected date range
df_filtered = df_filtered[(
    (df_filtered['date'] >= DATE_START.astype(np.datetime64))
   &(df_filtered['date'] <= DATE_END)
)]
#  4. Exclude selected models
df_filtered = df_filtered[~df_filtered['optimal_model_id'].isin(MODELS_TO_EXCLUDE)]
#  5. Aggregate data to desired cadence
if DATE_AGGREGATION == 'Daily':
    df_filtered['date_agg'] = df_filtered['date']
    df_agg = df_filtered.groupby(['date_agg', 'optimal_model_id'])[['spend', 'revenue']].agg([np.sum]).reset_index()
elif DATE_AGGREGATION == 'Weekly':
    df_filtered['date_agg'] = [dt - np.timedelta64(dt.isoweekday() - 1, 'D') for dt in df_filtered['date']]
    df_agg = df_filtered.groupby(['date_agg', 'optimal_model_id'])[['spend', 'revenue']].agg([np.sum]).reset_index()
elif DATE_AGGREGATION == 'Monthly':
    df_filtered['date_agg'] = [np.datetime64(f"{dt.year}-{dt.month:02}-01") for dt in df_filtered['date']]
    df_agg = df_filtered.groupby(['date_agg', 'optimal_model_id'])[['spend', 'revenue']].agg([np.sum]).reset_index()
elif DATE_AGGREGATION == 'Quarterly':
    df_filtered['date_agg'] = [np.datetime64(f"{dt.year}-{(((dt.month - 1) // 3) * 3) + 1:02}-01") for dt in df_filtered['date']]
    df_agg = df_filtered.groupby(['date_agg', 'optimal_model_id'])[['spend', 'revenue']].agg([np.sum]).reset_index()
else:
    pass

df_agg.columns = [f'{col1}_{col2}' for col1, col2 in df_agg.columns]
df_agg['RoAS'] = df_agg['revenue_sum'] / df_agg['spend_sum']

#st.dataframe(df_agg)



#st.info(':information_source:  here it is :shark: ::')
#st.warning(':warning:  watch out!')
# with st.container():
#     col1, col2, col3, col4, col5, col6 = st.columns((1,1,1,1,1,1))
#
#     with col1:
#         do_something = st.button('Click me!', ['a', 'b', 'c'])
#         st.caption("Use this to define the area of focus")
#
#     with col2:
#         choice = st.radio('One choice', ['a', 'b', 'c'])
#
#     with col3:
#         choice = st.selectbox('One choice', ['a', 'b', 'c'])
#         st.caption("Use this to define the area of focus")
#
#     with col4:
#         choices = st.multiselect('Pick many', ['a', 'b', 'c', 'd', 'e'])
#         st.caption("Use this to define the area of focus")
#
#     with col5:
#         start_date = st.date_input('Start date')
#         st.caption("Use this to define the area of focus")
#
#     with col6:
#         hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
#         st.caption("Use this to define the area of focus")

def format_number(number):
    if number >= 1000000000:
        return f'{number/1e9:.1f}b'
    elif number >= 50000000 and number < 1000000000:
        return f'{number/1e6:.0f}m'
    elif number >= 1000000 and number < 50000000:
        return f'{number/1e6:.1f}m'
    elif number >= 50000 and number < 1000000:
        return f'{number/1e3:.0f}k'
    elif number >= 1000 and number < 50000:
        return f'{number/1e3:.1f}k'
    elif number >= 50 and number < 1000:
        return f'{number:.0f}'
    elif number >= 10 and number < 50:
        return f'{number:.1f}'
    elif number >= 1 and number < 10:
        return f'{number:.2f}'
    elif number >=0 and number < 1:
        return f'{number:.2f}'
    else:
        return f'{number}'



st.header('RoAS')

st.subheader('At a glance')
col1, col2, col3, col4 = st.columns((1, 1, 1, 3))
with col1:
    st.metric(label="Spend", value=format_number(df_agg['spend_sum'].sum()), delta=34)
with col2:
    st.metric(label="Revenue", value=format_number(df_agg['revenue_sum'].sum()), delta=-43)
with col3:
    st.metric(label="RoAS", value=format_number(df_agg['revenue_sum'].sum() / df_agg['spend_sum'].sum()), delta=0.1)
with col4:
    pass


#st.subheader('RoAS Time Series')
col1, col2 = st.columns((2, 1))
with col1:
    st.subheader('Time series plot')

    fig = px.scatter(
        df_agg,  # .query("year==2007"),
        x="date_agg_",
        y="RoAS",
        # size="pop",
        color="optimal_model_id_",
        # hover_name="country",
        # log_x=True,
        opacity=0.5,
        size_max=60,
    )
    fig.update_layout(yaxis_range=[0, 15])

    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.info(':information_source:  click on models to add/remove from plot')

with col2:
    st.subheader('Tabular data')

    #st.dataframe(df_agg)
    df_agg_over_models = df_agg.groupby('date_agg_')[['spend_sum', 'revenue_sum']].mean()
    df_agg_over_models['RoAS'] = df_agg_over_models['revenue_sum'] / df_agg_over_models['spend_sum']

    #st.dataframe(df_agg_over_models)


    if DATE_AGGREGATION == 'Daily':
        df_agg_over_models['date_readable'] = [dt.strftime('%d-%b-%Y') for dt in df_agg_over_models.index]
    elif DATE_AGGREGATION == 'Weekly':
        df_agg_over_models['date_readable'] = [dt.strftime('%Y-W%W') for dt in df_agg_over_models.index]
    elif DATE_AGGREGATION == 'Monthly':
        df_agg_over_models['date_readable'] = [dt.strftime('%b-%Y') for dt in df_agg_over_models.index]
    elif DATE_AGGREGATION == 'Quarterly':
        df_agg_over_models['date_readable'] = [f"{dt.year}-Q{int((dt.month - 1) / 3) + 1}" for dt in df_agg_over_models.index]
    else:
        pass
    df_agg_over_models.set_index('date_readable', inplace=True)

    #do_something = st.button(f'Download {CADENCE} data (CSV)')

    if st.checkbox('Use dates as columns'):
        df_display = df_agg_over_models.T
    else:
        df_display = df_agg_over_models.copy(deep=True)

    st.dataframe(df_display)

    st.download_button(
        label=f"Export {DATE_AGGREGATION} data (CSV)",
        data=df_display.to_csv(index=True).encode('utf-8'),
        file_name=f'{DATE_AGGREGATION}-MMM-RoAS-data.csv',
        mime='text/csv',
    )


with st.container():
    col1, col2 = st.columns((2,1))

    with col1:
        with st.container():
            st.subheader('RoAS Time Series')
        with st.container():
            st.subheader('Raw data')
            with st.expander("See explanation"):
                st.write("""
                    The chart above shows some numbers I picked for you.
                    I rolled actual dice for these, so they're *guaranteed* to
                    be random.
                """)

            # do_something = st.button(f'Download {CADENCE} data')
            # if do_something:
            #     st.write(f'{CADENCE} data downloaded!')
            #     a = 1


        #import streamlit as st

        #df = px.data.gapminder()



st.header('Response curves')
st.caption("Use this to define the area of focus")
with st.container():
    col1, col2 = st.columns((3,2))

    with col1:
        st.subheader('CT.com model')

    with col2:
        st.subheader('B&M model')

st.header('Carry-over effect')
st.caption("Use this to define the area of focus")
with st.container():
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)
    st.pyplot(fig)

# # FUNCTION FOR AIRPORT MAPS
# def map(data, lat, lon, zoom):
#     st.write(
#         pdk.Deck(
#             map_style="mapbox://styles/mapbox/light-v9",
#             initial_view_state={
#                 "latitude": lat,
#                 "longitude": lon,
#                 "zoom": zoom,
#                 "pitch": 50,
#             },
#             layers=[
#                 pdk.Layer(
#                     "HexagonLayer",
#                     data=data,
#                     get_position=["lon", "lat"],
#                     radius=100,
#                     elevation_scale=4,
#                     elevation_range=[0, 1000],
#                     pickable=True,
#                     extruded=True,
#                 ),
#             ],
#         )
#     )
#
#
# # FILTER DATA FOR A SPECIFIC HOUR, CACHE
# @st.experimental_memo
# def filterdata(df, hour_selected):
#     return df[df["date/time"].dt.hour == hour_selected]
#
#
# # CALCULATE MIDPOINT FOR GIVEN SET OF DATA
# @st.experimental_memo
# def mpoint(lat, lon):
#     return (np.average(lat), np.average(lon))
#
#
# # FILTER DATA BY HOUR
# @st.experimental_memo
# def histdata(df, hr):
#     filtered = data[
#         (df["date/time"].dt.hour >= hr) & (df["date/time"].dt.hour < (hr + 1))
#     ]
#
#     hist = np.histogram(filtered["date/time"].dt.minute, bins=60, range=(0, 60))[0]
#
#     return pd.DataFrame({"minute": range(60), "pickups": hist})
#
#
# # STREAMLIT APP LAYOUT
# data = load_data()
#
# # LAYING OUT THE TOP SECTION OF THE APP
# row1_1, row1_2 = st.columns((2, 3))
#
# # SEE IF THERE'S A QUERY PARAM IN THE URL (e.g. ?pickup_hour=2)
# # THIS ALLOWS YOU TO PASS A STATEFUL URL TO SOMEONE WITH A SPECIFIC HOUR SELECTED,
# # E.G. https://share.streamlit.io/streamlit/demo-uber-nyc-pickups/main?pickup_hour=2
# if not st.session_state.get("url_synced", False):
#     try:
#         pickup_hour = int(st.experimental_get_query_params()["pickup_hour"][0])
#         st.session_state["pickup_hour"] = pickup_hour
#         st.session_state["url_synced"] = True
#     except KeyError:
#         pass
#
# # IF THE SLIDER CHANGES, UPDATE THE QUERY PARAM
# def update_query_params():
#     hour_selected = st.session_state["pickup_hour"]
#     st.experimental_set_query_params(pickup_hour=hour_selected)
#
#
# with row1_1:
#     st.title("NYC Uber Ridesharing Data")
#     hour_selected = st.slider(
#         "Select hour of pickup", 0, 23, key="pickup_hour", on_change=update_query_params
#     )
#
#
# with row1_2:
#     st.write(
#         """
#     ##
#     Examining how Uber pickups vary over time in New York City's and at its major regional airports.
#     By sliding the slider on the left you can view different slices of time and explore different transportation trends.
#     """
#     )
#
# # LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
# row2_1, row2_2, row2_3, row2_4 = st.columns((2, 1, 1, 1))
#
# # SETTING THE ZOOM LOCATIONS FOR THE AIRPORTS
# la_guardia = [40.7900, -73.8700]
# jfk = [40.6650, -73.7821]
# newark = [40.7090, -74.1805]
# zoom_level = 12
# midpoint = mpoint(data["lat"], data["lon"])
#
# with row2_1:
#     st.write(
#         f"""**All New York City from {hour_selected}:00 and {(hour_selected + 1) % 24}:00**"""
#     )
#     map(filterdata(data, hour_selected), midpoint[0], midpoint[1], 11)
#
# with row2_2:
#     st.write("**La Guardia Airport**")
#     map(filterdata(data, hour_selected), la_guardia[0], la_guardia[1], zoom_level)
#
# with row2_3:
#     st.write("**JFK Airport**")
#     map(filterdata(data, hour_selected), jfk[0], jfk[1], zoom_level)
#
# with row2_4:
#     st.write("**Newark Airport**")
#     map(filterdata(data, hour_selected), newark[0], newark[1], zoom_level)
#
# # CALCULATING DATA FOR THE HISTOGRAM
# chart_data = histdata(data, hour_selected)
#
# # LAYING OUT THE HISTOGRAM SECTION
# st.write(
#     f"""**Breakdown of rides per minute between {hour_selected}:00 and {(hour_selected + 1) % 24}:00**"""
# )
#
# st.altair_chart(
#     alt.Chart(chart_data)
#     .mark_area(
#         interpolate="step-after",
#     )
#     .encode(
#         x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
#         y=alt.Y("pickups:Q"),
#         tooltip=["minute", "pickups"],
#     )
#     .configure_mark(opacity=0.2, color="red"),
#     use_container_width=True,
# )
