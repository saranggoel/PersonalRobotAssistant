from flask import Flask, request
import sys

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def main():

    if request.method == 'POST':


        return

    else:
        return '''
    <!doctype html>
        <title>A Personal Robot Assistant Controlled with Eye-Tracking Technology</title>
         '''


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

