from oauth2client.client import GoogleCredentials
import requests
import json
import fire

def run(project, model_name, model_version, token=None):
    if token is None:
        token = GoogleCredentials.get_application_default().get_access_token().access_token

    api = 'https://ml.googleapis.com/v1/projects/{}/models/{}/versions/{}:predict' \
             .format(project, model_name, model_version)
    headers = {'Authorization': 'Bearer ' + token }
    data = {'instances':
      [
          {
            'pickuplon': -73.885262,
            'pickuplat': 40.773008,
            'dropofflon': -73.987232,
            'dropofflat': 40.732403,
            'passengers': 2,
            'dayofweek': 'Wed',
            'hourofday' : 10,
            'key': 1
          }
      ]
    }

    response = requests.post(api, json=data, headers=headers)
    print(response.content)

if __name__ == '__main__':
    fire.Fire(run)
