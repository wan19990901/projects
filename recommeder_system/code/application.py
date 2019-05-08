from flask import Flask,render_template,url_for,request
import os
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
from plotly import tools
from plotly.offline import plot
import plotly.graph_objs as go
from flask import Markup
import flask
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
application = Flask(__name__)
app = application
movie = pd.read_csv('data/movie.csv')
data_movie = movie['Combined']
movie.replace({'Title': 'A Night to Remember  '},'The Witcher 3: Wild Hunt Launch',inplace=True)

@app.route('/')
def home():
    choice = []
    action = movie[movie['Genre'].str.contains('Action') ].sort_values(by='scores')[::-1][:10]
    comedy = movie[movie['Genre'].str.contains('Comedy') ].sort_values(by='scores')[::-1][:10]
    docu = movie[movie['Genre'].str.contains('Document') ].sort_values(by='scores')[::-1][:10]
    mystery = movie[movie['Genre'].str.contains('Mystery') ].sort_values(by='scores')[::-1][:10]
    roman = movie[movie['Genre'].str.contains('Romance') ].sort_values(by='scores')[::-1][:10]
    drama = movie[movie['Genre'].str.contains('Drama') ].sort_values(by='scores')[::-1][:10]
    scifi = movie[movie['Genre'].str.contains('SciFi') ].sort_values(by='scores')[::-1][:10]
    classic = movie[movie['Genre'].str.contains('Classic') ].sort_values(by='scores')[::-1][:10]
    All = movie.sort_values(by='scores')[::-1][:10]
    choice = [action,comedy,docu,mystery,roman,drama,scifi,classic,All]
    return render_template("index.html",choice = choice)
# server = dash.Dash(__name__, server = app, url_base_pathname='/Action/' )


@app.route('/test',methods=['POST','GET'])
def test():
    if request.method == 'POST':
        
        with open('model.pkl','rb') as fid:
            model = pickle.load(fid)
        with open('model_recommend.pkl','rb') as fid:
            vec = pickle.load(fid)
		
        data_movie = movie['Combined']
        message = request.form['message']
        
        data = vec.transform([message])
		
        result = model.kneighbors(data)
        info = []
        labels = []
        idx = []
        scores = []
        for i in result[1][0]:
            t = movie.iloc[i]['Title']
            if(pd.isna(t)):
                continue
            else:
                j = np.argwhere(result[1][0] == i)[0][0]
            if(result[0][0][j] != 1):
                l = [t,movie.iloc[i]]
                scores.append(movie['scores'].iloc[i])
                info.append([movie['scores'].iloc[i],movie['Genre'].iloc[i],l,movie['url'].iloc[i]])
                idx.append(j)
                labels.append(t)
    values = result[0][0][idx]
    values_query = 1/1000 * (10 - values)**4
    values_score = values_query * 0.2 + np.array(scores) * 0.8
    try:
        if(values[0] == 1):
            return render_template('test.html',prediction = 'bad')
    except IndexError:
        return render_template('test.html',prediction = 'bad') 
    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1','#A9D9B1']
    trace = go.Pie(labels=labels, values=values_query,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)),
                           domain=dict(x=[0,0.5]))
    trace_scores = go.Pie(labels=labels, values=values_score,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)),
                           domain=dict(x=[0.5,1]))
    data = [trace,trace_scores]
    ann1 = dict(font=dict(size=20),
            showarrow=False,
            text='Pure search scores',
            # Specify text position (place text in a hole of pie)
            x=0.20,
            y=1.2
            )
    ann2 = dict(font=dict(size=20),
            showarrow=False,
            text='Combined rating and search scores',
            # Specify text position (place text in a hole of pie)
            x=0.95,
            y=1.2
            )
    layout = go.Layout(title ='Two results',
                   annotations=[ann1,ann2],
                   )
    fig = go.Figure(data=data,layout=layout)
    my_plot = plot(fig, output_type='div')

    return render_template('test.html',prediction = info,mydiv = Markup(my_plot))

@app.route('/action')
def action():
    df_action = movie[movie['Genre'].str.contains('Action') ]
    info = []
    info.append(len(df_action))
    info.append(df_action['Cast 1'].value_counts().index[0])
    info.append(df_action['Director 1'].value_counts().index[0])
    info.append(df_action['Studio'].value_counts().index[0])
    info.append([df_action['scores'].sort_values()[::-1].iloc[2],df_action[df_action['scores'] == df_action['scores'].sort_values()[::-1].iloc[2]]])
    info.append([df_action['scores'].min(),df_action[df_action['scores'] == df_action['scores'].min()]])
    fig = tools.make_subplots(rows=2, cols=2,subplot_titles = ('Most Popular Actors','Most popular director','Most popular studio','Scores Distribution'))
    actor = go.Bar(
                y=df_action['Cast 1'].value_counts()[:10],
                x=df_action['Cast 1'].value_counts()[:10].index[:10],
                text=df_action['Cast 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(234,125,92)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
                name = 'Actor',
                opacity=0.6
            )
    director = go.Bar(
                y=df_action['Director 1'].value_counts()[:10],
                x=df_action['Director 1'].value_counts()[:10].index[:10],
                text=df_action['Director 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(119,174,103)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Director',
                opacity=0.6
            )
    studio = go.Bar(
                y=df_action['Studio'].value_counts()[:10],
                x=df_action['Studio'].value_counts()[:10].index[:10],
                text=df_action['Studio'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(204,106,118)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Studio',
                opacity=0.6
            )
    score = go.Histogram(
                x=df_action['scores'],
                xbins=dict(
                start=0,
                end=10,
                size=0.5
            ),
                marker=dict(
                    color='rgb(103,162,178)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Score',
                opacity=0.6
            )

    fig.append_trace(actor, 1, 1)
    fig.append_trace(director, 1, 2)
    fig.append_trace(studio, 2, 1)
    fig.append_trace(score, 2, 2)
    fig['layout'].update(height=1600, width=1000, title='Information about Action movie')
    my_plot = plot(fig, output_type='div')
    return render_template('action.html',info = info,mydiv = Markup(my_plot))
@app.route('/comedy')
def comedy():
    df_action = movie[movie['Genre'].str.contains('Comedy') ]
    info = []
    info.append(len(df_action))
    info.append(df_action['Cast 1'].value_counts().index[0])
    info.append(df_action['Director 1'].value_counts().index[0])
    info.append(df_action['Studio'].value_counts().index[0])
    info.append([df_action['scores'].max(),df_action[df_action['scores'] == df_action['scores'].max()]])
    info.append([df_action['scores'].min(),df_action[df_action['scores'] == df_action['scores'].min()]])
    fig = tools.make_subplots(rows=2, cols=2,subplot_titles = ('Most Popular Actors','Most popular director','Most popular studio','Scores Distribution'))
    actor = go.Bar(
                y=df_action['Cast 1'].value_counts()[:10],
                x=df_action['Cast 1'].value_counts()[:10].index[:10],
                text=df_action['Cast 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(234,125,92)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
                name = 'Actor',
                opacity=0.6
            )
    director = go.Bar(
                y=df_action['Director 1'].value_counts()[:10],
                x=df_action['Director 1'].value_counts()[:10].index[:10],
                text=df_action['Director 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(119,174,103)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Director',
                opacity=0.6
            )
    studio = go.Bar(
                y=df_action['Studio'].value_counts()[:10],
                x=df_action['Studio'].value_counts()[:10].index[:10],
                text=df_action['Studio'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(204,106,118)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Studio',
                opacity=0.6
            )
    score = go.Histogram(
                x=df_action['scores'],
                xbins=dict(
                start=0,
                end=10,
                size=0.5
            ),
                marker=dict(
                    color='rgb(103,162,178)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Score',
                opacity=0.6
            )

    fig.append_trace(actor, 1, 1)
    fig.append_trace(director, 1, 2)
    fig.append_trace(studio, 2, 1)
    fig.append_trace(score, 2, 2)
    fig['layout'].update(height=1600, width=1000, title='Information about Comedy movie')
    my_plot = plot(fig, output_type='div')
    return render_template('comedy.html',info = info,mydiv = Markup(my_plot))
@app.route('/docu')
def docu():
    df_action = movie[movie['Genre'].str.contains('Documentary') ]
    df_action.replace('  ','Not Available',inplace = True)
    info = []
    info.append(len(df_action))
    info.append(df_action['Cast 1'].value_counts().index[0])
    info.append(df_action['Director 1'].value_counts().index[0])
    info.append(df_action['Studio'].value_counts().index[0])
    info.append([df_action['scores'].max(),df_action[df_action['scores'] == df_action['scores'].max()]])
    info.append([df_action['scores'].min(),df_action[df_action['scores'] == df_action['scores'].min()]])
    fig = tools.make_subplots(rows=2, cols=2,subplot_titles = ('Most Popular Actors','Most popular director','Most popular studio','Scores Distribution'))
    actor = go.Bar(
                y=df_action['Cast 1'].value_counts()[:10],
                x=df_action['Cast 1'].value_counts()[:10].index[:10],
                text=df_action['Cast 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(234,125,92)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
                name = 'Actor',
                opacity=0.6
            )
    director = go.Bar(
                y=df_action['Director 1'].value_counts()[:10],
                x=df_action['Director 1'].value_counts()[:10].index[:10],
                text=df_action['Director 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(119,174,103)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Director',
                opacity=0.6
            )
    studio = go.Bar(
                y=df_action['Studio'].value_counts()[:10],
                x=df_action['Studio'].value_counts()[:10].index[:10],
                text=df_action['Studio'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(204,106,118)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Studio',
                opacity=0.6
            )
    score = go.Histogram(
                x=df_action['scores'],
                xbins=dict(
                start=0,
                end=10,
                size=0.5
            ),
                marker=dict(
                    color='rgb(103,162,178)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Score',
                opacity=0.6
            )

    fig.append_trace(actor, 1, 1)
    fig.append_trace(director, 1, 2)
    fig.append_trace(studio, 2, 1)
    fig.append_trace(score, 2, 2)
    fig['layout'].update(height=1600, width=1000, title='Information about documentarys movie')
    my_plot = plot(fig, output_type='div')
    return render_template('docu.html',info = info,mydiv = Markup(my_plot))
@app.route('/drama')
def drama():
    df_action = movie[movie['Genre'].str.contains('Drama') ]
    df_action.replace('  ','Not Available',inplace = True)
    info = []
    info.append(len(df_action))
    info.append(df_action['Cast 1'].value_counts().index[0])
    info.append(df_action['Director 1'].value_counts().index[0])
    info.append(df_action['Studio'].value_counts().index[0])
    info.append([df_action['scores'].max(),df_action[df_action['scores'] == df_action['scores'].max()]])
    info.append([df_action['scores'].min(),df_action[df_action['scores'] == df_action['scores'].min()]])
    fig = tools.make_subplots(rows=2, cols=2,subplot_titles = ('Most Popular Actors','Most popular director','Most popular studio','Scores Distribution'))
    actor = go.Bar(
                y=df_action['Cast 1'].value_counts()[:10],
                x=df_action['Cast 1'].value_counts()[:10].index[:10],
                text=df_action['Cast 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(234,125,92)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
                name = 'Actor',
                opacity=0.6
            )
    director = go.Bar(
                y=df_action['Director 1'].value_counts()[:10],
                x=df_action['Director 1'].value_counts()[:10].index[:10],
                text=df_action['Director 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(119,174,103)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Director',
                opacity=0.6
            )
    studio = go.Bar(
                y=df_action['Studio'].value_counts()[:10],
                x=df_action['Studio'].value_counts()[:10].index[:10],
                text=df_action['Studio'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(204,106,118)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Studio',
                opacity=0.6
            )
    score = go.Histogram(
                x=df_action['scores'],
                xbins=dict(
                start=0,
                end=10,
                size=0.5
            ),
                marker=dict(
                    color='rgb(103,162,178)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Score',
                opacity=0.6
            )

    fig.append_trace(actor, 1, 1)
    fig.append_trace(director, 1, 2)
    fig.append_trace(studio, 2, 1)
    fig.append_trace(score, 2, 2)
    fig['layout'].update(height=1600, width=1000, title='Information about Comedy movie')
    my_plot = plot(fig, output_type='div')
    return render_template('drama.html',info = info,mydiv = Markup(my_plot))
@app.route('/classic')
def classic():
    df_action = movie[movie['Genre'].str.contains('Classic') ]
    df_action.replace('  ','Not Available',inplace = True)
    info = []
    info.append(len(df_action))
    info.append(df_action['Cast 1'].value_counts().index[0])
    info.append(df_action['Director 1'].value_counts().index[0])
    info.append(df_action['Studio'].value_counts().index[0])
    info.append([df_action['scores'].sort_values()[::-1].iloc[2],df_action[df_action['scores'] == df_action['scores'].sort_values()[::-1].iloc[2]]])
    info.append([df_action['scores'].min(),df_action[df_action['scores'] == df_action['scores'].min()]])
    fig = tools.make_subplots(rows=2, cols=2,subplot_titles = ('Most Popular Actors','Most popular director','Most popular studio','Scores Distribution'))
    actor = go.Bar(
                y=df_action['Cast 1'].value_counts()[:10],
                x=df_action['Cast 1'].value_counts()[:10].index[:10],
                text=df_action['Cast 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(234,125,92)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
                name = 'Actor',
                opacity=0.6
            )
    director = go.Bar(
                y=df_action['Director 1'].value_counts()[:10],
                x=df_action['Director 1'].value_counts()[:10].index[:10],
                text=df_action['Director 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(119,174,103)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Director',
                opacity=0.6
            )
    studio = go.Bar(
                y=df_action['Studio'].value_counts()[:10],
                x=df_action['Studio'].value_counts()[:10].index[:10],
                text=df_action['Studio'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(204,106,118)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Studio',
                opacity=0.6
            )
    score = go.Histogram(
                x=df_action['scores'],
                xbins=dict(
                start=0,
                end=10,
                size=0.5
            ),
                marker=dict(
                    color='rgb(103,162,178)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Score',
                opacity=0.6
            )

    fig.append_trace(actor, 1, 1)
    fig.append_trace(director, 1, 2)
    fig.append_trace(studio, 2, 1)
    fig.append_trace(score, 2, 2)
    fig['layout'].update(height=1600, width=1000, title='Information about Comedy movie')
    my_plot = plot(fig, output_type='div')
    return render_template('classic.html',info = info,mydiv = Markup(my_plot))
@app.route('/romance')
def romance():
    df_action = movie[movie['Genre'].str.contains('Romance') ]
    df_action.replace('  ','Not Available',inplace = True)
    info = []
    info.append(len(df_action))
    info.append(df_action['Cast 1'].value_counts().index[0])
    info.append(df_action['Director 1'].value_counts().index[0])
    info.append(df_action['Studio'].value_counts().index[0])
    info.append([df_action['scores'].max(),df_action[df_action['scores'] == df_action['scores'].max()]])
    info.append([df_action['scores'].min(),df_action[df_action['scores'] == df_action['scores'].min()]])
    fig = tools.make_subplots(rows=2, cols=2,subplot_titles = ('Most Popular Actors','Most popular director','Most popular studio','Scores Distribution'))
    actor = go.Bar(
                y=df_action['Cast 1'].value_counts()[:10],
                x=df_action['Cast 1'].value_counts()[:10].index[:10],
                text=df_action['Cast 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(234,125,92)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
                name = 'Actor',
                opacity=0.6
            )
    director = go.Bar(
                y=df_action['Director 1'].value_counts()[:10],
                x=df_action['Director 1'].value_counts()[:10].index[:10],
                text=df_action['Director 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(119,174,103)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Director',
                opacity=0.6
            )
    studio = go.Bar(
                y=df_action['Studio'].value_counts()[:10],
                x=df_action['Studio'].value_counts()[:10].index[:10],
                text=df_action['Studio'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(204,106,118)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Studio',
                opacity=0.6
            )
    score = go.Histogram(
                x=df_action['scores'],
                xbins=dict(
                start=0,
                end=10,
                size=0.5
            ),
                marker=dict(
                    color='rgb(103,162,178)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Score',
                opacity=0.6
            )

    fig.append_trace(actor, 1, 1)
    fig.append_trace(director, 1, 2)
    fig.append_trace(studio, 2, 1)
    fig.append_trace(score, 2, 2)
    fig['layout'].update(height=1600, width=1000, title='Information about Comedy movie')
    my_plot = plot(fig, output_type='div')
    return render_template('romance.html',info = info,mydiv = Markup(my_plot))
@app.route('/scifi')
def scifi():
    df_action = movie[movie['Genre'].str.contains('SciFi') ]
    df_action.replace('  ','Not Available',inplace = True)
    info = []
    info.append(len(df_action))
    info.append(df_action['Cast 1'].value_counts().index[0])
    info.append(df_action['Director 1'].value_counts().index[0])
    info.append(df_action['Studio'].value_counts().index[0])
    info.append([df_action['scores'].max(),df_action[df_action['scores'] == df_action['scores'].max()]])
    info.append([df_action['scores'].min(),df_action[df_action['scores'] == df_action['scores'].min()]])
    fig = tools.make_subplots(rows=2, cols=2,subplot_titles = ('Most Popular Actors','Most popular director','Most popular studio','Scores Distribution'))
    actor = go.Bar(
                y=df_action['Cast 1'].value_counts()[:10],
                x=df_action['Cast 1'].value_counts()[:10].index[:10],
                text=df_action['Cast 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(234,125,92)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
                name = 'Actor',
                opacity=0.6
            )
    director = go.Bar(
                y=df_action['Director 1'].value_counts()[:10],
                x=df_action['Director 1'].value_counts()[:10].index[:10],
                text=df_action['Director 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(119,174,103)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Director',
                opacity=0.6
            )
    studio = go.Bar(
                y=df_action['Studio'].value_counts()[:10],
                x=df_action['Studio'].value_counts()[:10].index[:10],
                text=df_action['Studio'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(204,106,118)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Studio',
                opacity=0.6
            )
    score = go.Histogram(
                x=df_action['scores'],
                xbins=dict(
                start=0,
                end=10,
                size=0.5
            ),
                marker=dict(
                    color='rgb(103,162,178)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Score',
                opacity=0.6
            )

    fig.append_trace(actor, 1, 1)
    fig.append_trace(director, 1, 2)
    fig.append_trace(studio, 2, 1)
    fig.append_trace(score, 2, 2)
    fig['layout'].update(height=1600, width=1000, title='Information about Comedy movie')
    my_plot = plot(fig, output_type='div')
    return render_template('scifi.html',info = info,mydiv = Markup(my_plot))
@app.route('/mystery')
def mystery():
    df_action = movie[movie['Genre'].str.contains('Mystery') ]
    df_action.replace('  ','Not Available',inplace = True)
    info = []
    info.append(len(df_action))
    info.append(df_action['Cast 1'].value_counts().index[0])
    info.append(df_action['Director 1'].value_counts().index[0])
    info.append(df_action['Studio'].value_counts().index[0])
    info.append([df_action['scores'].max(),df_action[df_action['scores'] == df_action['scores'].max()]])
    info.append([df_action['scores'].min(),df_action[df_action['scores'] == df_action['scores'].min()]])
    fig = tools.make_subplots(rows=2, cols=2,subplot_titles = ('Most Popular Actors','Most popular director','Most popular studio','Scores Distribution'))
    actor = go.Bar(
                y=df_action['Cast 1'].value_counts()[:10],
                x=df_action['Cast 1'].value_counts()[:10].index[:10],
                text=df_action['Cast 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(234,125,92)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
                name = 'Actor',
                opacity=0.6
            )
    director = go.Bar(
                y=df_action['Director 1'].value_counts()[:10],
                x=df_action['Director 1'].value_counts()[:10].index[:10],
                text=df_action['Director 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(119,174,103)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Director',
                opacity=0.6
            )
    studio = go.Bar(
                y=df_action['Studio'].value_counts()[:10],
                x=df_action['Studio'].value_counts()[:10].index[:10],
                text=df_action['Studio'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(204,106,118)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Studio',
                opacity=0.6
            )
    score = go.Histogram(
                x=df_action['scores'],
                xbins=dict(
                start=0,
                end=10,
                size=0.5
            ),
                marker=dict(
                    color='rgb(103,162,178)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Score',
                opacity=0.6
            )

    fig.append_trace(actor, 1, 1)
    fig.append_trace(director, 1, 2)
    fig.append_trace(studio, 2, 1)
    fig.append_trace(score, 2, 2)
    fig['layout'].update(height=1600, width=1000, title='Information about Comedy movie')
    my_plot = plot(fig, output_type='div')
    return render_template('mystery.html',info = info,mydiv = Markup(my_plot))
@app.route('/all')
def all():
    df_action = movie
    df_action.replace('  ','Not Available',inplace = True)
    info = []
    info.append(len(df_action))
    info.append(df_action['Cast 1'].value_counts().index[0])
    info.append(df_action['Director 1'].value_counts().index[0])
    info.append(df_action['Studio'].value_counts().index[0])
    info.append([df_action['scores'].max(),df_action[df_action['scores'] == df_action['scores'].max()]])
    info.append([df_action['scores'].min(),df_action[df_action['scores'] == df_action['scores'].min()]])
    fig = tools.make_subplots(rows=2, cols=2,subplot_titles = ('Most Popular Actors','Most popular director','Most popular studio','Scores Distribution'))
    actor = go.Bar(
                y=df_action['Cast 1'].value_counts()[:10],
                x=df_action['Cast 1'].value_counts()[:10].index[:10],
                text=df_action['Cast 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(234,125,92)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
                name = 'Actor',
                opacity=0.6
            )
    director = go.Bar(
                y=df_action['Director 1'].value_counts()[:10],
                x=df_action['Director 1'].value_counts()[:10].index[:10],
                text=df_action['Director 1'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(119,174,103)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Director',
                opacity=0.6
            )
    studio = go.Bar(
                y=df_action['Studio'].value_counts()[:10],
                x=df_action['Studio'].value_counts()[:10].index[:10],
                text=df_action['Studio'].value_counts()[:10],
                textposition = 'auto',
                marker=dict(
                    color='rgb(204,106,118)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Studio',
                opacity=0.6
            )
    score = go.Histogram(
                x=df_action['scores'],
                xbins=dict(
                start=0,
                end=10,
                size=0.5
            ),
                marker=dict(
                    color='rgb(103,162,178)',
                    line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                ),
        name = 'Score',
                opacity=0.6
            )

    fig.append_trace(actor, 1, 1)
    fig.append_trace(director, 1, 2)
    fig.append_trace(studio, 2, 1)
    fig.append_trace(score, 2, 2)
    fig['layout'].update(height=1600, width=1000, title='Information about Comedy movie')
    my_plot = plot(fig, output_type='div')
    return render_template('Overall.html',info = info,mydiv = Markup(my_plot))
if __name__ == "__main__":
    app.run()
    # jobs for next time: A dynamic graph in text.html based both rating and distances, Provide some general movie recommendation in index.html page