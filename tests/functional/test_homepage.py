from urllib import response
import pytest
import os, os.path
from flask import Flask, request

@pytest.fixture()
def test_homepage():
    flask_app = Flask(__name__)

    with flask_app.test_client() as test_client:
        response = test_client.get('/')
        assert response.status_code == '200'
        assert b'According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?' in response.data