import pandas as pd
import json


@pd.api.extensions.register_series_accessor("json")
class JsonForCadAccessor:
    """
    This extension to the pandas Series class will enable it to add a json data inside a cell (or update the current
    json stored there through .json namespace.
    """
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj  # type: pd.Series

    @staticmethod
    def _validate(obj):
        # verify it is a series, not a dataframe
        if not isinstance(obj, pd.Series):
            raise AttributeError('pd.json only works with series. For processing DataFrames you can split each column '
                                 'to a different Series')
        pass

    @staticmethod
    def add_to_json(x: str, input_json: str):
        # load input json to a dict
        try:
            _input_json = json.loads(input_json)
        except ValueError as e:
            raise ValueError('input json string cannot be decoded as json for: ' + input_json)

        # load existing json to a dict
        try:
            _x = json.loads(x)
            # if the variable is parsed, but it is not a dict, replace it with an empty dict
            if not isinstance(_x, dict):
                _x = dict()
        except:
            # if can't parse json, replace it with an empty json
            _x = dict()

        # merge dictionaries
        _x.update(_input_json)

        return json.dumps(_x)

    def join(self, input_json: pd.Series, inplace=False):
        """
        This method adds a json string to the json data stored in a cell. If the data in the cell is not json, the new
        data will be overwritten
        :param inplace: if True, the pandas object will be changed directly (not by return value)
        :param input_json: the json string of data desired to be stored in the json structure inside the pandas cell
        :return:
        """
        df = pd.DataFrame([], columns=['A', 'B'])
        df['A'] = self._obj
        df['B'] = input_json

        df['C'] = df.apply(lambda x: self.add_to_json(x['A'], x['B']), axis=1)

        if inplace:
            self._obj.loc[:] = df['C']

        return df['C']


# a = pd.DataFrame({'n': [1, 2, 3], 'j': ['{"IsCardiovascularDeath": false}', [], '{"IsCardiovascularDeath": true}']})
# b = pd.DataFrame({'j2': ['w', 'e', 'r']})
#
# q = a['j'].json.join(b['j2'].apply(lambda x: json.dumps({'TEST': x})), inplace=True)
# # q = a['j'].json.join(input_json=pd.Series(['{"y": true}', '{"2y": false}', '{"y": true}']))
# print(q)
# print(a['j'])
