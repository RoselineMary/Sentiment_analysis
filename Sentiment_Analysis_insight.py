# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:50:10 2021

@author: Roseline
"""
# Importing the libraries
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = "Sentiment Analysis with Insights"

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

def load_model():
    global pickle_model
    global vocab
    global scrappedReviews
    global values1
    global labels1
    
    
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    balancedReviews = pd.read_csv("balanced_reviews.csv")
    balancedReviews.dropna(inplace = True)
    balancedReviews = balancedReviews[balancedReviews['overall'] != 3]
    balancedReviews['Positivity'] = np.where(balancedReviews['overall'] > 3, 1, 0 )
    values1=[sum(balancedReviews['Positivity']), len(balancedReviews)-sum(balancedReviews['Positivity'])]
    labels1=["Postive", 'Negative']
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)
    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)
        
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

def create_app_ui():
    global project_name
    main_layout = dbc.Container(
        [   html.H1(id = 'heading', children = project_name,
                     style = {
                "textAlign" : "center",
                "color" : "#FFFFFF"
                
                }),
         html.Hr(),
        
        dbc.Row([
            dbc.Col(
            
        dbc.Jumbotron(
                [
                    dcc.Graph(id='Pie_Chart',
                           figure=go.Figure(
                               data=[go.Pie(labels=labels1,
                                            values=values1,
                                            marker=dict(
                   colors=['rgb(153,255,153)', 'rgb(255,153,153)'])
                                            )],
                               layout=go.Layout(
                                   title='Overall review analysis from the data')
                           )),
                                        ],
                className = 'text-center'
                ),
        ), 
         dbc.Col(           
        dbc.Jumbotron(
                [
                    dbc.Container([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews.reviews],
                    value = scrappedReviews.reviews[0],
                    style = {'margin-bottom': '30px' , "width" : "100%", "white-space": "normal"}
                    
                )
                       ],
                        style = {'padding-left': '2px', 'padding-right': '95px',"width": "120%"}
                        ),
                    html.Div(id = 'result1'),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-3", id = 'button1', style = {'width': '100px'}),
                    dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", value = 'Honeslty speaking the colour is very bad and the metal of the earning is of bad quality',
                                 style = {'height': '110px'}),
                    html.Div(id = 'result'),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
                  ],
                className = 'text-center1'
                ),   
        ),        
        ])
        ],
        style={"height": "100vh","background-color": "#d3d3d3" , "width" : "100%"},
        )   
    return main_layout

@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

@app.callback(
    Output('result1', 'children'),
    [
    Input('button1', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
def main():
    global app
    global project_name
    load_model()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    app = None
    project_name = None
if __name__ == '__main__':
    main()






