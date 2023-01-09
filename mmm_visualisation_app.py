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
from datetime import datetime, timedelta
import pydeck as pdk
#import streamlit_nested_layout
import streamlit as st
from matplotlib import pyplot as plt, colors, dates
import plotly.express as px
import plotly.figure_factory as ff

# fig.patch.set_facecolor('#FFFFFF')
# fig.patch.set_alpha(0.0)
#
# ax.patch.set_facecolor('#FFFFFF')
# ax.patch.set_alpha(0.0)

plt.style.use('matplotlib_housestyle_dark.mplstyle')
# plt.rcParams['grid.alpha']=0.5
# plt.rcParams['grid.color']='#DDDDDD'
# plt.rcParams['grid.linestyle']='-'
# plt.rcParams['grid.linewidth']='1'
# plt.rcParams['figure.facecolor']='#dd44aa'#'#FFFFFF'
# plt.rcParams['axes.facecolor']='#1122ee'#'#FFFFFF'
# #plt.rcParams['axesfacecolor']='red'
# #plt.rcParams['axesfacecolor']='red'
# #plt.rcParams['axesfacecolor']='red'











np.random.seed(42)

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


def simplify_list_of_date_strings(list_of_timestamps):
    '''
    Simplify list of date strings so that years and months are not
    repeated in consecutive values

    For example, given the following list of dates:

        11 Nov 1999
        21 Nov 1999
        01 Dec 1999
        11 Dec 1999
        21 Dec 1999
        31 Dec 1999
        11 Jan 2000
        21 Jan 2000

    This would be simplified to:

        11 Nov 1999
        21 ^   ^
         1 Dec ^
        11 ^   ^
        21 ^   ^
        31 ^   ^
        11 Jan 2000
        21 ^   ^

    Parameters
    ----------
        list_of_timestamps : array (of timestamps)
            Timestamps

    Returns
    -------
        simplified_list_of_date_strings : array (of str)
            Simplified list of date strings where year-parts and
            month-parts are not repeated in consecutive values
    '''

    # Keep first value as it is
    simplified_list_of_date_strings = [
        f'{list_of_timestamps[0].day}' + datetime.strftime(list_of_timestamps[0], '\n%b\n%Y'), ]

    # Loop through timestamp
    for i in range(1, len(list_of_timestamps)):

        current = list_of_timestamps[i]
        previous = list_of_timestamps[i - 1]

        # If year of current date is the same as year of previous date
        # AND month of current date is the same as month of previous date
        #   e.g., 1 Jan 1999 and 2 Jan 1999  ->  '2'
        # Then only show day-part
        if (current.year == previous.year) and (current.month == previous.month):
            simplified_date_string = f'{current.day}\n'  # datetime.strftime(current, '%d\n')

        # If year of current date is the same as year of previous date
        # BUT month of current date is different to month of previous date
        #   e.g., 31 Jan 1999 and 1 Feb 1999  ->  '1 Feb'
        # Then only show day-part and month-part
        elif (current.year == previous.year) and (current.month != previous.month):
            simplified_date_string = f'{current.day}' + datetime.strftime(current,
                                                                          '\n%b\n')  # datetime.strftime(current, '%d\n%b\n')

        # If year of current date is different to year of previous date
        # AND month of current date is different to month of previous date
        # Then show whole date
        #   e.g., 31 Dec 1999 and 1 Jan 2000  ->  '1 Jan 2000'
        elif (current.year != previous.year) and (current.month != previous.month):
            simplified_date_string = f'{current.day}' + datetime.strftime(current,
                                                                          '\n%b\n%Y')  # datetime.strftime(current, '%d\n%b\n%Y')

        # Catch any exceptions (for whatever reason)
        else:
            simplified_date_string = 'unformatted'

        simplified_list_of_date_strings.append(simplified_date_string)

    return simplified_list_of_date_strings

# Page title
st.set_page_config(layout="wide", page_title="CT MMM Results - Channel one-pager", page_icon=":m::m::m:")

st.warning(":warning: For a PoC, this has been done on my public GitHub profile, but don't worry, this is not real data")

st.warning(":warning: Plots can be made interactive using `plotly` or `streamlit`, but at the expense of flexibility in how data can be visualised")


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
                for optimal_model_id in [f'1_111_{i}' for i in range(1, 50)]:
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

    DATE_AGGREGATION = st.radio('Date aggregation', ('Daily', 'Weekly', 'Monthly', 'Quarterly'), 2)

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

col1, col2, col3, col4, col5, col6 = st.columns((1, 1, 1, 1, 1, 2))
with col1:
    st.metric(label="Spend", value=format_number(df_agg['spend_sum'].sum()), delta=34)
with col2:
    st.metric(label="Attributed\nRevenue", value=format_number(df_agg['revenue_sum'].sum()), delta=-43)
with col3:
    st.metric(label="RoAS", value=format_number(df_agg['revenue_sum'].sum() / df_agg['spend_sum'].sum()), delta=0.1)
with col4:
    st.metric(label="Share of spend", value='13%')
with col5:
    st.metric(label="Share of effect", value='18%')
with col6:
    pass


#st.subheader('RoAS Time Series')
col1, col2 = st.columns((2, 1))
with col1:
    st.subheader('Time series plot')

    # fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    # ax.scatter(
    #     x=df_agg['date_agg_'],
    #     y=df_agg['RoAS'],
    #     s=25,
    #     marker='s',
    #     color='#6E2132',
    #     alpha=0.05,
    #     zorder=3,
    #     label=None,
    # )
    # date_form = dates.DateFormatter("%d %b\n%Y")
    # ax.xaxis.set_major_formatter(date_form)
    # ax.set_xticklabels(
    #     simplify_list_of_date_strings([datetime(1970, 1, 1) + timedelta(days=i) for i in ax.get_xticks()]),
    #     fontstyle='italic')
    # ax.grid(zorder=0)
    # ax.set_ylim(bottom=0.0)
    # ax.legend(loc='lower left')
    # ax.set_ylabel('RoAS', fontstyle='oblique')
    # fig.patch.set_alpha(0.0)
    # ax.patch.set_alpha(0.0)
    # st.pyplot(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    df_agg_intervals = df_agg.groupby('date_agg_')[['spend_sum', 'revenue_sum']].agg([np.min, q05, q10, q20, q32, np.median, q68, q80, q90, q95, np.max])
    df_agg_intervals.columns = [f'{col1}_{col2}' for col1, col2 in df_agg_intervals.columns]
    for agg in ['amin', 'q05', 'q10', 'q20', 'q32', 'median', 'q68', 'q80', 'q90', 'q95', 'amax']:
        df_agg_intervals[f'RoAS_{agg}'] = df_agg_intervals[f'revenue_sum_{agg}'] / df_agg_intervals[f'spend_sum_{agg}']
    df_agg_intervals.reset_index(inplace=True)
    #st.dataframe(df_agg_intervals)

    for low, high in [('amin', 'amax'), ('q05', 'q95'), ('q10', 'q90'), ('q20', 'q80'), ('q32', 'q68')]:
        l = df_agg_intervals[f'RoAS_{low}']
        h = df_agg_intervals[f'RoAS_{high}']
        ax.fill_between(
            df_agg_intervals['date_agg_'],
            l,
            h,
            color  = '#6E2132',
            #linestyle=None,
            linewidth=0.0,
            alpha  = 0.2,
            # label  = f'range',
            zorder=3,
            #label='predicted'
        )
    ax.plot(
        df_agg_intervals['date_agg_'],
        df_agg_intervals['RoAS_median'],
        # s=25,
        # marker='s',
        color='#6E2132',
        ls = '-',
        linewidth='2',
        # alpha=0.05,
        zorder=3,
        # label=None,
    )

    date_form = dates.DateFormatter("%d %b\n%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xticklabels(
        simplify_list_of_date_strings([datetime(1970, 1, 1) + timedelta(days=i) for i in ax.get_xticks()]),
        fontstyle='italic')
    ax.grid(zorder=0)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc='lower left')
    ax.set_ylabel('RoAS', fontstyle='oblique')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    st.pyplot(fig)


    # fig = px.scatter(
    #     df_agg,  # .query("year==2007"),
    #     x="date_agg_",
    #     y="RoAS",
    #     # size="pop",
    #     color="optimal_model_id_",
    #     # hover_name="country",
    #     # log_x=True,
    #     opacity=0.5,
    #     size_max=60,
    # )
    # fig.update_layout(yaxis_range=[0, 15])
    #
    # # Use the Streamlit theme.
    # # This is the default. So you can also omit the theme argument.
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    #
    # st.info(':information_source:  click on models to add/remove from plot')





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

    st.dataframe(df_display, height=250)

    st.download_button(
        label=f"Export {DATE_AGGREGATION} data (CSV)",
        data=df_display.to_csv(index=True).encode('utf-8'),
        file_name=f'{DATE_AGGREGATION}-MMM-RoAS-data.csv',
        mime='text/csv',
    )




# with st.container():
#     col1, col2 = st.columns((2,1))
#
#     with col1:
#         with st.container():
#             st.subheader('RoAS Time Series')
#         with st.container():
#             st.subheader('Raw data')
#             with st.expander("See explanation"):
#                 st.write("""
#                     The chart above shows some numbers I picked for you.
#                     I rolled actual dice for these, so they're *guaranteed* to
#                     be random.
#                 """)

            # do_something = st.button(f'Download {CADENCE} data')
            # if do_something:
            #     st.write(f'{CADENCE} data downloaded!')
            #     a = 1


        #import streamlit as st

        #df = px.data.gapminder()

# TODO: remove gridlines

# TODO: remove ff plots and go back to plotly express
@st.experimental_singleton
def generate_simulated_response_curves():
    data = []
    investment = np.linspace(0,1000,101)
    for model_id in ['CT.com', 'B&M']:
        for channel in ['YouTube', 'TikTok', 'Bing', 'TV']:
            for optimal_model_id in [f'1_111_{i}' for i in range(1, 50)]:
                n = np.random.gamma(10.0, 0.02, 1)
                A = np.random.gamma(50, 5, 1)
                for x in investment:

                    return_ = A * (x**n)
                    data.append([model_id, optimal_model_id, channel, x, return_[0]])

    df = pd.DataFrame(data=data, columns=['model_id', 'optimal_model_id', 'channel', 'investment', 'return'])
    df['RoAS'] = df['return'] / df['investment']
    df = df.sort_values(by=['model_id', 'optimal_model_id', 'channel', 'investment'])
    return df

df_response_curves = generate_simulated_response_curves()

# Given options in sidebar, filter and aggregate the data accordingly
#  1. Keep selected model
df_response_curves = df_response_curves[df_response_curves['model_id']==MODEL]
#  2. Keep selected channel
df_response_curves = df_response_curves[df_response_curves['channel']==CHANNEL]
#  3. Exclude selected models
df_response_curves = df_response_curves[~df_response_curves['optimal_model_id'].isin(MODELS_TO_EXCLUDE)]


st.header('Response curves')
with st.container():
    #RESPONSE_CURVE_METRIC = st.radio('Metric to display in response curves', ('return', 'RoAS'), 0)

    col1, col2, col3 = st.columns((1, 1, 2))

    with col1:
        RESPONSE_CURVE_METRIC = st.radio('Metric to display in response curves', ('return', 'RoAS'), 0)

    with col2:
        INVESTMENT_LEVEL = st.select_slider("Investment level", options=df_response_curves['investment'].unique(),
                                            value=500, key="x")

    with col3:
        pass

#st.caption("Use this to define the area of focus")
with st.container():

    col1, col2 = st.columns((2,2))

    with col1:
        st.subheader('Return-investment curves')

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

        for optimal_model_id in df_response_curves['optimal_model_id']:

            df_ = df_response_curves[df_response_curves['optimal_model_id']==optimal_model_id]

            ax.plot(
                df_['investment'],
                df_[RESPONSE_CURVE_METRIC],
                #s=25,
                #marker='s',
                color='#6E2132',
                #alpha=0.05,
                zorder=3,
                #label=None,
            )
            # date_form = dates.DateFormatter("%d %b\n%Y")
            # ax.xaxis.set_major_formatter(date_form)
            # ax.set_xticklabels(
            #     simplify_list_of_date_strings([datetime(1970, 1, 1) + timedelta(days=i) for i in ax.get_xticks()]),
            #     fontstyle='italic')
            # ax.grid(zorder=0)
            # ax.set_ylim(bottom=0.0)
            # ax.legend(loc='lower left')
            # ax.set_ylabel('RoAS', fontstyle='oblique')
        ax.plot(
            [INVESTMENT_LEVEL, INVESTMENT_LEVEL],
            [0, ax.get_ylim()[1]],
            ls='--',
            color='#E09894',
            zorder=4,
        )

        ax.set_xlabel('Investment', fontstyle='oblique')
        ax.set_ylabel(RESPONSE_CURVE_METRIC, fontstyle='oblique')
        ax.grid(zorder=0)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        st.pyplot(fig)


        # fig = px.line(
        #     df_response_curves,
        #     x="investment",
        #     y=RESPONSE_CURVE_METRIC,
        #     # size="pop",
        #     color="optimal_model_id",
        #     # hover_name="country",
        #     # log_x=True,
        #     #opacity=0.5,
        #     #size_max=60,
        # )
        # #fig.update_layout(yaxis_range=[0, 15])

        # # Use the Streamlit theme.
        # # This is the default. So you can also omit the theme argument.
        # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        #
        # st.info(':information_source:  click on models to add/remove from plot')

    with col2:
        st.subheader('Probable returns')
        #
        df_investment_level = df_response_curves[df_response_curves['investment']==INVESTMENT_LEVEL]


        # the histogram of the data
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        n, bins, patches = ax.hist(df_investment_level[RESPONSE_CURVE_METRIC], 10, facecolor='#6E2132', density=True, alpha=0.75, zorder=3)
        #print(n, bins)

        # fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        # mu, sigma = 100, 15
        # x = mu + sigma * np.random.randn(10000)
        # x = df_investment_level[RESPONSE_CURVE_METRIC]
        # n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
        # print(n, bins)

        ax.set_xlabel(RESPONSE_CURVE_METRIC, fontstyle='oblique')
        # ax.set_ylabel('Probability')
        # plt.title('Histogram of IQ')
        # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        # plt.xlim(40, 160)
        # plt.ylim(0, 0.03)
        ax.plot(
            [INVESTMENT_LEVEL, INVESTMENT_LEVEL],
            [0, ax.get_ylim()[1]],
            ls='--',
            color='#E09894',
            zorder=4
        )

        ax.grid(zorder=0)
        ax.set_ylim(bottom=0.0)
        ax.set_yticks([])
        #ax.legend(loc='lower left')
        #ax.set_ylabel('RoAS', fontstyle='oblique')
        # plt.show()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        st.pyplot(fig)


        # # TODO: add bar plot here
        # hist1, bins = np.histogram(df_investment_level[RESPONSE_CURVE_METRIC], bins=20)
        # # df_hist1 = pd.DataFrame(data=hist), columns=['value', 'bin'])
        # # st.write(df_hist1)
        # st.bar_chart(data=hist1)

        #fig = ff.create_distplot([df_investment_level[RESPONSE_CURVE_METRIC]], group_labels=['xx'], bin_size=df_investment_level[RESPONSE_CURVE_METRIC].max()/20)
        # #fig = px.histogram(df, x="return", y="return")
        # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        # #fig.show()

col1, col2, col3, col4 = st.columns((1, 1, 1, 3))

with col1:
    st.metric(label="Optimum RoAS at:", value='£1,030')

with col2:
    st.metric(label="Optimum RoI at:", value='£500')

with col3:
    st.metric(label="Maximum profit at:", value='£2,850')

with col4:
        pass


@st.experimental_singleton
def generate_simulated_adstock_decay_rates():
    data = []
    investment = np.linspace(0,1000,101)
    for model_id in ['CT.com', 'B&M']:
        for channel in ['YouTube', 'TikTok', 'Bing', 'TV']:
            adstock_decay_rate = 1.0 / len(channel)
            for optimal_model_id in [f'1_111_{i}' for i in range(1, 50)]:
                theta = np.random.gamma(adstock_decay_rate*10, 0.1, 1)
                data.append([model_id, optimal_model_id, channel, theta[0]])

    df = pd.DataFrame(data=data, columns=['model_id', 'optimal_model_id', 'channel', 'theta'])
    df = df.sort_values(by=['model_id', 'optimal_model_id', 'channel'])
    return df

df_adstock_decay_rates = generate_simulated_adstock_decay_rates()

# Given options in sidebar, filter and aggregate the data accordingly
#  1. Keep selected model
df_adstock_decay_rates = df_adstock_decay_rates[df_adstock_decay_rates['model_id']==MODEL]
#  2. Keep selected channel
df_adstock_decay_rates = df_adstock_decay_rates[df_adstock_decay_rates['channel']==CHANNEL]
#  3. Exclude selected models
df_adstock_decay_rates = df_adstock_decay_rates[~df_adstock_decay_rates['optimal_model_id'].isin(MODELS_TO_EXCLUDE)]

#st.dataframe(df_adstock_decay_rates)

st.header('Carry-over effect')

col1, col2 = st.columns((1, 1))

with col1:
    st.subheader('Adstock decay')

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    n, bins, patches = ax.hist(df_adstock_decay_rates['theta'], 10, facecolor='#6E2132', density=True,
                               alpha=0.75, zorder=3)
    # print(n, bins)

    # fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    # mu, sigma = 100, 15
    # x = mu + sigma * np.random.randn(10000)
    # x = df_investment_level[RESPONSE_CURVE_METRIC]
    # n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
    # print(n, bins)

    ax.set_xlabel('Carry-over effect', fontstyle='oblique')
    # ax.set_ylabel('Probability')
    # plt.title('Histogram of IQ')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    ax.grid(zorder=0)
    ax.set_ylim(bottom=0.0)
    ax.set_yticks([])
    # ax.legend(loc='lower left')
    # ax.set_ylabel('RoAS', fontstyle='oblique')
    # plt.show()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    st.pyplot(fig)


    # # TODO: add bar plot here
    # hist, bins = np.histogram(df_adstock_decay_rates['theta'], bins=np.linspace(0, 1, 21))
    # #df_hist1 = pd.DataFrame(data=hist), columns=['value', 'bin'])
    # #st.write(df_hist1)
    # st.bar_chart(data=hist)

    #print(hist)

    # fig = ff.create_distplot([df_adstock_decay_rates['theta']], group_labels=['x3'], bin_size=0.05)
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)


with col2:
    st.subheader('Strength of effect with time')

    days = np.linspace(1, 14, 14)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    for theta in df_adstock_decay_rates['theta']:
        adstock = theta ** days

        ax.plot(
            days,
            adstock,
            # s=25,
            # marker='s',
            color='#6E2132',
            # alpha=0.05,
            zorder=3,
            # label=None,
        )
        # date_form = dates.DateFormatter("%d %b\n%Y")
        # ax.xaxis.set_major_formatter(date_form)
        # ax.set_xticklabels(
        #     simplify_list_of_date_strings([datetime(1970, 1, 1) + timedelta(days=i) for i in ax.get_xticks()]),
        #     fontstyle='italic')
        # ax.grid(zorder=0)
        # ax.set_ylim(bottom=0.0)
        # ax.legend(loc='lower left')
        # ax.set_ylabel('RoAS', fontstyle='oblique')
    ax.set_xlabel('Elapsed days', fontstyle='oblique')
    ax.set_ylabel('Adstock', fontstyle='oblique')
    ax.grid(zorder=0)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    st.pyplot(fig)


    # theta = df_adstock_decay_rates['theta'].mean()
    # df = pd.DataFrame(data=zip(days, carry_over_effect), columns=['elapsed_days', 'relative_effect'])
    # #st.dataframe(df)
    #
    #
    #
    # # df['id'] = 'a'
    # #
    # fig = px.line(
    #     df,
    #     x='elapsed_days',
    #     y='relative_effect',
    #     # size="pop",
    #     #color="id",
    #     # hover_name="country",
    #     # log_x=True,
    #     # opacity=0.5,
    #     # size_max=60,
    # )
    # # fig.update_layout(yaxis_range=[0, 15])
    #
    # # Use the Streamlit theme.
    # # This is the default. So you can also omit the theme argument.
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    #
    # INVESTMENT_LEVEL = st.select_slider("Investment level", options=df_response_curves['investment'].unique(), key="x2")
    #
    # df = df_response_curves[df_response_curves['investment'] == INVESTMENT_LEVEL]
    #
    #
    # fig = ff.create_distplot([df[RESPONSE_CURVE_METRIC]], group_labels=['x'],
    #                          bin_size=df[RESPONSE_CURVE_METRIC].max() / 20)
    # # fig = px.histogram(df, x="return", y="return")
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    # # fig.show()




# TODO: you can make pyplot charts interactive!

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
