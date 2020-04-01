#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from flask import Flask, render_template, request
from werkzeug import secure_filename
import logging
from logging import Formatter, FileHandler
from forms import *
from models.svhnNeturalNetwork import SVHNEstimator, SVMEstimator
import os

#----------------------------------------------------------------------------#
# App Config. 
# For Ref. pip3 freeze > requirments.txt
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')

#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/')
def home():
    return render_template('pages/placeholder.home.html')


@app.route('/about')
def about():
    return render_template('pages/placeholder.about.html')

# @app.route('/train')
# def train():
#     return render_template('pages/placeholder.about.html')

@app.route('/train')
def train():
    estimator = SVHNEstimator()
    estimator.loadDataset()
    estimator.build()
    history = estimator.fit(100)
    estimator.saveTrainedModel()
    return render_template('pages/placeholder.training.html', training_results=history.history)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictForm()
    # if request.method == 'POST':
    # else:
    results = None
    sfname = None
    if request.method == 'POST':
        f = request.files['photo']
        filename = secure_filename(form.photo.data.filename)
        sfname = 'uploaded-images/'+str(filename)
        f.save(sfname)


        estimator = SVHNEstimator()
        estimator.loadTrainedModel()
        results = estimator.predict(sfname)
    else:
        filename = None
   
    return render_template('forms/predict.html', form=form, filename=filename, results = results, img=sfname)

 

@app.route('/login')
def login():
    form = LoginForm(request.form)
    return render_template('forms/login.html', form=form)


@app.route('/register')
def register():
    form = RegisterForm(request.form)
    return render_template('forms/register.html', form=form)


@app.route('/forgot')
def forgot():
    form = ForgotForm(request.form)
    return render_template('forms/forgot.html', form=form)

# Error handlers.


@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run()

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''
