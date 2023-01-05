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
import streamlit as st
from matplotlib import pyplot as plt

# Page title
st.set_page_config(layout="wide", page_title="CT MMM Results - Channel one-pager", page_icon=":circled_m::circled_m::circled_m:")


# LOAD DATA ONCE
@st.experimental_singleton
def load_data():
    from datetime import datetime, timedelta
    import pandas as pd
    data = []
    for model_id in ['UK CT.com', 'UK B&M']:
        for optimal_model_id in ['1_100_1', '2_323_2', '4_444_1', '5_455_5', '6_666_6', '7_777_7', '8_888_8',
                                 '9_099_0']:
            for date in [(datetime(2022, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(0, 365)]:
                for channel in ['YouTube', 'TikTok', 'Bing', 'TV']:
                    spend = np.random.gamma(20, 5, 1)[0]
                    revenue = np.random.gamma(25, 5, 1)[0]
                    data.append([model_id, optimal_model_id, channel, date, spend, revenue])
    df = pd.DataFrame(data=data, columns=['model_id', 'optimal_model_id', 'channel', 'date', 'spend', 'revenue'])
    df = df.sort_values(by=['model_id', 'optimal_model_id', 'channel', 'date'])
    # df.to_csv('~/Desktop/roas-toy-data.csv', index=False)
    # df = pd.read_csv(
    #     "simulated_data.csv",
    #     parse_dates=["date"],  # set as datetime instead of converting after the fact
    # )

    return df

df = load_data()


# TODO: make plots using `df`



st.header('Parameters')
st.caption("Use this to define the area of focus")
st.info(':information_source:  here it is :shark: ::')
st.warning(':warning:  watch out!')
with st.container():
    col1, col2, col3, col4, col5, col6 = st.columns((1,1,1,1,1,1))

    with col1:
        do_something = st.button('Click me!', ['a', 'b', 'c'])
        st.caption("Use this to define the area of focus")

    with col2:
        choice = st.radio('One choice', ['a', 'b', 'c'])

    with col3:
        choice = st.selectbox('One choice', ['a', 'b', 'c'])
        st.caption("Use this to define the area of focus")

    with col4:
        choices = st.multiselect('Pick many', ['a', 'b', 'c', 'd', 'e'])
        st.caption("Use this to define the area of focus")

    with col5:
        start_date = st.date_input('Start date')
        st.caption("Use this to define the area of focus")

    with col6:
        hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
        st.caption("Use this to define the area of focus")


st.header('RoAS')
st.caption("Use this to define the area of focus")
with st.container():
    col1, col2 = st.columns((3,2))

    with col1:
        st.write("Lorem lorem lorem")
        if st.checkbox('I will show something...'):
            st.write('Here I am!')

    with col2:
        st.subheader('Raw data')
        with st.expander("See explanation"):
            st.write("""
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
            """)

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
