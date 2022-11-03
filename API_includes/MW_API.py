# Adapted from original source: https://ttv.microworkers.com/assets/RestClient/MW_API.py.txt?v25 last checked 04.04.2021

import requests


class MW_API:

    def __init__(self, api_key, api_url='https://api.microworkers.com'):
        self.api_key = api_key
        self.api_url = api_url

    def do_request(self, method='', action='', params={}, files={}):
        method = method.lower()
        if not method in ['get', 'put', 'post', 'delete']:
            raise Exception('Method: "' + method + '" is not supported')

        headers = {'MicroworkersApiKey': self.api_key}

        if method == 'get':
            r = requests.get(self.api_url + action, params=params, headers=headers)
        if method == 'put':
            r = requests.put(self.api_url + action, json=params, headers=headers)
        if method == 'post':
            r = requests.post(self.api_url + action, files=files, json=params, headers=headers)
        if method == 'delete':
            r = requests.delete(self.api_url + action, data=params, headers=headers)
        if method == 'patch':
            r = requests.patch(self.api_url + action, json=params, headers=headers)

        # print(r.text)

        if r.status_code == requests.codes.ok:
            try:
                value = r.json()
                return {'type': 'json', 'value': value}
            except ValueError as e:
                print(e)
                return {'type': 'body', 'value': r.content}

        try:
            r.raise_for_status()
        except Exception as e:
            print(e)
            return {'type': 'error', 'value': e}
