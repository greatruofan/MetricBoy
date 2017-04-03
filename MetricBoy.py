import matplotlib
import requests
import json
import pandas as pd
from pandas.io.json import json_normalize
import datetime as dt
import numpy as np
from urllib.parse import quote
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import argparse
import sys
from argparse import RawTextHelpFormatter
import decimal

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM



class MetricBoy:

    LOGIN_URL = 'https://argus-ws.data.sfdc.net/argusws/auth/login'
    CREDENTIAL_FILE_NAME = 'login.json'


    def __init__(self):
    #def __init__(self):

        self.username, self.password = self._read_credential(MetricBoy.CREDENTIAL_FILE_NAME)
        self.session = self.login(self.username, self.password)

        self.plot_type_func_map = {'line': self.line,
                                   'bar': self.bar,
                                   'barh': self.barh,
                                   'hist': self.hist,
                                   'box': self.box,
                                   'kde': self.kde,
                                   'area': self.area,
                                   'pie': self.pie,
                                   'scatter': self.scatter
                                   }

        #The minus sign at the beginning of metric expression will confuse argsparse with other optional argument
        #i.e. "-30m:argus.custom:query.latency{timeWindow=within_24_hrs}:avg" need a whitespace prefix
        #Instead " -30m:argus.custom:query.latency{timeWindow=within_24_hrs}:avg"
        for i, arg in enumerate(sys.argv):
            if (arg[0] == '-') and arg[1].isdigit(): sys.argv[i] = ' ' + arg
        self.parse_metric_args()
        args = self.parser.parse_args()
        # for i, arg in enumerate(sys.argv):

        # # if self.session is None :
        # #     self.session = self.login(args)

        args.func(args)
        print(args)


        #self.session = self.login_args(args)

    def login_args(self, args):
        json_data = {
            'username': args.username,
            'password': args.password
        }

        with open(CREDENTIAL_FILE_NAME, mode='w', encoding='utf-8') as f:
            json.dump(json_data, f)

    def _read_credential(self, credential_file):

        import os.path
        if not os.path.isfile(credential_file):
            print("Credential file doesn't exist, please login first!")

        with open(credential_file) as json_data:
            data = json.load(json_data)
            username = data["username"]
            password = data["password"]
        return username, password

    def metric_args(self, args):

        save_plot_file_name = args.save_plot_file if args.save_plot_file is None else args.save_plot_file.name
        self.metric(args.metric_exp[0],#because metric_exp argument accepts a list
                    args.plot_type,
                    args.table,
                    args.save_plot,
                    args.save_table,
                    save_plot_file_name,#save csv need an exact name string, instead of a file
                    args.save_table_file)

    def dashboard_args(self, args):
        save_plot_file_name = args.save_plot_file if args.save_plot_file is None else args.save_plot_file.name
        self.dashboard(args.metric_exp,
                       args.metric_file,
                       args.merge,
                       args.table,
                       args.save_plot,
                       args.save_table,
                       save_plot_file_name,#save csv need an exact name string, instead of a file
                       args.save_table_file)

    def predicate_args(self, args):
        save_plot_file_name = args.save_plot_file if args.save_plot_file is None else args.save_plot_file.name
        self.predicate(args.metric_exp[0],#because metric_exp argument accepts a list
                       args.epoch,
                       args.table,
                       args.save_plot,
                       args.save_table,
                       save_plot_file_name,#save csv need an exact name string, instead of a file
                       args.save_table_file)


    def parse_metric_args(self):
        self.shared_parser = argparse.ArgumentParser(add_help=False)
        self.shared_parser.add_argument('--metric_exp', metavar='METRIC_EXP', type=str, action='append',
                                        help='Metric expression')
        self.shared_parser.add_argument('--table', action='store_true', help='display metric data table')
        self.shared_parser.add_argument('--save_plot', action='store_true', help='save the result as a plot file')
        self.shared_parser.add_argument('--save_plot_file', type=argparse.FileType('w'),
                                        help='specify file name of saved plot file')
        self.shared_parser.add_argument('--save_table', action='store_true', help='save the result as a table file')
        self.shared_parser.add_argument('--save_table_file', type=argparse.FileType('w'),
                                        help='specify file name of saved table file')

        self.parser = argparse.ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter,
                                              prog="MetricBoy",
                                              parents=[self.shared_parser],
                                              usage="""MetricBoy <command> [<args>]""",
                                              description="""
            ##############################################################################
            ## MetricBoy is a client for querying, plotting and predicating metric data ##
            ##############################################################################

            The main MetricBoy commands are:
                metric         Query data for a single metric
                dashboard      Query data for a list of metrics
                predicate      Machine learning(RNN) against your query metric
                               Predictation or Anomaly detection based on real-time data

            For any command above, you can either display the data in a table or plot the
            data with a given chart type, you can also save the table and the plot, please
            refer to optional arguments description
            ---------------------------------------------------------------------------------

            Examples:

                metric
                =============================================================================
                Example 1. plot a metric with scatter type, available plot type are
                'line', 'bar', 'barh', 'hist', 'box', 'kde', 'area', 'pie', 'scatter'

                $ python MetricBoy.py --metric_exp
                "-30m:argus.custom:query.latency{timeWindow=within_24_hrs}:avg"
                metric --plot_type ‘scatter'

                Example 2. display datapoints table for a metric,
                and plot a metric with default line type,argus.custom:query.latency.csv
                argus.custom:query.latency.png

                $ python MetricBoy.py --metric_exp
                "-30m:argus.custom:query.latency{timeWindow=within_24_hrs}:avg"
                --table metric


                Example 3. same as Example 2, also save the table and plot as files with
                metric name as default file name, which is argus.custom:query.latency.csv
                and argus.custom:query.latency.png for this example

                $ python MetricBoy.py --metric_exp
                "-30m:argus.custom:query.latency{timeWindow=within_24_hrs}:avg"
                --table --save_table --save_plot metric

                Example 4. same as Example 3, but specify file names for saving

                $ python MetricBoy.py --metric_exp
                "-30m:argus.custom:query.latency{timeWindow=within_24_hrs}:avg"
                --table --save_table --save_plot
                --save_table_file 'try.csv' --save_plot_file 'try.png' metric


                dashboard
                ===========================================================================
                Example 1. create dashboard for metrics

                $ python MetricBoy.py --metric_exp "-7d:-0d:argus.core:datapoint.reads:sum"
                --metric_exp "-7d:-0d:argus.core:datapoint.writes:sum" dashboard


                Example 2. same as Example 1, but ploy metrics in one chart

                $ python MetricBoy.py --metric_exp "-7d:-0d:argus.core:datapoint.reads:sum"
                --metric_exp "-7d:-0d:argus.core:datapoint.writes:sum" dashboard -merge


                Example 3. same as Example 2, but also display table for datatpoints of all
                metrics

                $ python MetricBoy.py --metric_exp "-7d:-0d:argus.core:datapoint.reads:sum"
                --metric_exp "-7d:-0d:argus.core:datapoint.writes:sum"
                --table dashboard --merge


                Example 4. same as Example 2, but also save table and plot with default file
                name which is concatenation of all metrics names, for this example the file
                names are, argus.core:datapoint.readsargus.core:datapoint.writes_dashboard.csv
                and argus.core:datapoint.readsargus.core:datapoint.writes_dashboard.png

                $ python MetricBoy.py --metric_exp "-7d:-0d:argus.core:datapoint.reads:sum"
                --metric_exp "-7d:-0d:argus.core:datapoint.writes:sum"
                --save_table --save_plot dashboard --merge


                Example 5. same as Example 4, but save files with specifying file names

                $ python MetricBoy.py --metric_exp "-7d:-0d:argus.core:datapoint.reads:sum"
                --metric_exp "-7d:-0d:argus.core:datapoint.writes:sum"
                --save_table --save_plot
                --save_table_file 'dashboard.csv' --save_plot_file 'dashboard.png'
                dashboard --merge


                Example 6. plot dashboard by reading metrics from a input file
                $ python MetricBoy.py dashboard --metric_file "/Users/rzhang/py3/dashboard_metrics"


                Example 7. plot dashboard for both cmd input metrics and input file
                $ python MetricBoy.py --metric_exp "-7d:-0d:argus.core:datapoint.reads:sum"
                dashboard --metric_file "/Users/rzhang/py3/dashboard_metrics" --merge


                predicate
                ===========================================================================
                Example 7. RNN against 6000 datapoints with epoch 3000

                $ python MetricBoy.py --metric_exp
                "-6000m:argus.custom:query.latency{timeWindow=within_24_hrs}:avg"
                predicate --epoch 3000
            --------------------------------------------------------------------------------

            Warnings: The accuracy of predication based on the epoch time you set and
            the dataset size of metric datapoints you query.
            The bigger the epoch value is, the more accurate the result is. Same thing happens
            to your datapoints set size

            Warnings: To run predicate, the dataset of datapoints should be big enough, otherwise
            tensor cannot be created

            Warnings: Before any actions, login is required
          """
                                              )
        self.parser.add_argument('--help', '-h', action=self._HelpAction, help='help for help if you need some help')

        self.subparsers = self.parser.add_subparsers(title='MetricBoy commands',
                                                     description='MetricBoy provides the following commands')

        self.parser_login = self.subparsers.add_parser('login',
                                                       help='login command for user login')
        self.parser_login.add_argument('--username', help='username')
        self.parser_login.add_argument('--password', help='password')
        self.parser_login.set_defaults(func=self.login_args)
        # self.parser_login.add_argument('--help, -h', action=self._HelpAction, help='help for help if you need some help')

        self.parser_metric = self.subparsers.add_parser('metric', help='metric command for query one single metric')
        # self.parser_metric.add_argument('--metric', help='metric help')
        self.parser_metric.add_argument('--plot_type', type=str,
                                        choices=['line', 'bar', 'barh', 'hist', 'box', 'kde', 'area', 'pie', 'scatter'],
                                        default='line',
                                        help="chart type for plotting, please choose one from ['line', 'bar', 'barh', 'hist', 'box', 'kde', 'area', 'pie', 'scatter']")
        self.parser_metric.set_defaults(func=self.metric_args)

        self.parser_dashboard = self.subparsers.add_parser('dashboard', help='dashboard command for query metrics')
        # self.parser_dashboard.add_argument('--dashboard')
        self.parser_dashboard.add_argument('--metric_file', type=argparse.FileType('rb'),#('--metric_file', type=argparse.FileType('wb', 0),
                                           help='input file where metric expressions stored')
        self.parser_dashboard.add_argument('--merge', action='store_true',
                                           help='if set True, metrics will be merged in one table/chart')
        self.parser_dashboard.set_defaults(func=self.dashboard_args)

        self.parser_predicate = self.subparsers.add_parser('predicate',
                                                           help='predicate command for predicating metric data')
        # self.parser_predicate.add_argument('--predicate')
        self.parser_predicate.add_argument('--epoch', type=int, default=100, required=True,
                                           help='epoch configuration for recurrent neural network')
        self.parser_predicate.set_defaults(func=self.predicate_args)



    def login(self, username, password):

        '''Log into argus and return a live session'''

        payload = {'username':username, 'password':password}
        session = requests.Session()
        r = session.post(self.LOGIN_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload).encode('utf-8'))
        return session


    def _generate_metric_url(self, quote_metric_exp):
        return 'https://argus-ws.data.sfdc.net/argusws/metrics?expression={}'.format(quote_metric_exp)

    def _generate_metric_df(self, metric_exp):

        '''Generate a dataframe for a given metric expression'''

        rsp = self.session.get(self._generate_metric_url(quote(metric_exp, safe='')))
            #'https://argus-ws.data.sfdc.net/argusws/metrics?expression=-3000m:argus.custom:query.latency%7BtimeWindow%3Dwithin_24_hrs%7D:avg')

        json_data = json.loads(rsp.content.decode("utf-8"))
        #print(json_data)
        # df = pd.DataFrame(json_data)
        # df.fillna(0)

        # convert original dataframe to a formatted dataframe will be used
        dp_data = json_normalize(json_data)
        # deal with missing data, otherwise df doesn't work
        dp_data.fillna(0)
        original_df = pd.DataFrame(dp_data)

        #make label a class field which used by plotting pie and scatter
        self.metric_label = "{}:{}".format(original_df.scope[0], original_df.metric[0])
        filtered_df = original_df.filter(regex=("datapoints."))
        df = pd.DataFrame(filtered_df.T)
        df.columns = [self.metric_label]

        # can't convert to datetime in index, have a new column timestamp to hold it
        df['timestamp'] = df.index
        df['timestamp'] = df['timestamp'].str.extract('(\d+)', expand=True).astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        df.index = df['timestamp']

        # delete column name timestamp in index, delete timestamp column
        del df.index.name
        del df['timestamp']

        df = df.astype(float)

        return df

    ###########################  Metric ########################
    '''ax = df.plot()
fig = ax.get_figure()
fig.savefig('asdf.png')'''
    def metric(self, metric_exp, plot_type = 'line', table=False, save_plot=False,
               save_table=False, save_plot_name='metric.png', save_table_name='metric.csv'):
        # if args.plot_type:
        #     plot_type = args.plot_type
        plot_type_func = self.plot_type_func_map[plot_type]

        df = self._generate_metric_df(metric_exp)

        #plot = plot_type_func(df)

        if table:
            print(df)

        if save_table:
            if save_table_name is None:
                save_table_name = self.metric_label+'.csv'

            df.to_csv(save_table_name)

        ax =plot_type_func(df) #plot(df)

        if save_plot:
            if save_plot_name is None:
                save_plot_name = self.metric_label + '.png'

            fig = ax[0].get_figure() if plot_type=='scatter' else ax.get_figure()

            fig.savefig(save_plot_name)

        return df, ax

    def _plot_metric_df(plot_fn):
        def _plot_metric_df_wrapper(self, df):
            # df = _generate_metric_df(self, metric_exp)
            ax = plot_fn(self, df)
            plt.show()
            return ax

        return _plot_metric_df_wrapper

    #plot function need to return axesSubPlot object for saving
    @_plot_metric_df
    def line(self, df):
        return df.plot()


    @_plot_metric_df
    def bar(self, df):
        return df.plot(kind='bar')

    @_plot_metric_df
    def barh(self, df):
        return df.plot(kind='barh', stacked=True)

    @_plot_metric_df
    def hist(self, df):
        return df.plot(kind='hist')

    @_plot_metric_df
    def box(self, df):
        return df.plot(kind='box')

    @_plot_metric_df
    def kde(self, df):
        return df.plot(kind='kde')

    @_plot_metric_df
    def area(self, df):
        return df.plot(kind='area')

    @_plot_metric_df
    def pie(self, df):
        return df.plot(kind='pie', y=self.metric_label)
        # working pie chart with plt
        #plt.pie(df, labels=df[self.metric_label])

    @_plot_metric_df
    def scatter(self, df):
        # working scatter chart with plt
        return plt.plot_date(df.index, df[self.metric_label])

    ###########################  Dashboard ######################

    def dashboard(self, metric_exp_list=[], metric_file=None, merge = False,table=False, save_plot=False,
               save_table=False, save_plot_name='dashboard.png', save_table_name='dashboard.csv'):
        #Tricky: metric_file is BufferedReader, it return file's contents in byte
        if metric_file:
            parse_bufferedReader_metrics = metric_file.read().decode("utf-8")# convert content from byte to str
            metric_list_in_file = [s.strip() for s in parse_bufferedReader_metrics.splitlines()]

            if metric_exp_list is None:
                metric_exp_list = metric_list_in_file
            else:
                metric_exp_list.extend(metric_list_in_file)

            #print(metric_exp_list)

        # df_list = [self._generate_metric_df(metric_exp_list) for metric in metric_exp_list]
        df_list=[]
        label_list=[]
        for metric in metric_exp_list:
            df_list.append(self._generate_metric_df(metric))
            label_list.append(self.metric_label)





        if not merge:
            [single_df.plot() for single_df in df_list]
            plt.show()
            return

        merged_df = pd.concat(df_list, axis=1, join='outer')
        merged_ax = merged_df.plot()
        plt.show()

        if table:
            print(merged_df)

        if save_plot:
            if save_plot_name is None:
                save_plot_name = ''.join(label_list) + '_dashboard.png'

            fig = merged_ax.get_figure()
            fig.savefig(save_plot_name)

        if save_table:

            if save_table_name is None:
                save_table_name = ''.join(label_list) + '_dashboard.csv'

            merged_df.to_csv(save_table_name)







    #################    Prediction    #################
    IN_OUT_NEURONS = 1
    HIDDEN_NEURONS = 30

    def predicate(self, metric_exp, epoch=3000, table=False, save_plot=False,
               save_table=False, save_plot_name='predicted.png', save_table_name='predicted.csv'):

        #generate df from metric_exp and start training
        df = self._generate_metric_df(metric_exp)

        #scale down data for training purpose
        scale_down = self._scale_df_by(0.1)
        input_data = scale_down(df)
        print(input_data)
        print(self._calculate_scale_exponent(input_data))

        #start training

        import time
        start_time = time.time()

        predicted, y_test, rmse = self._rnn_train_df(input_data, epoch)

        print("--- %s seconds ---" % (time.time() - start_time))

        print(predicted)
        #scale up data after training
        scale_up = self._scale_df_by(10)
        predicted = scale_up(pd.DataFrame(predicted,columns=['predicted']))
        y_test = scale_up(pd.DataFrame(y_test,columns=['test']))

        print("Predicted data size is :{}".format(len(predicted)))
        print("RMSE is :{}".format(rmse))

        #plot individually and plot together
        df_result = pd.merge(predicted, y_test, left_index=True, right_index=True)#, suffixes=['_predicate', '_test'])

        predicted.plot()
        y_test.plot()
        df_result.plot()
        plt.show()

        if table:
            print(df_result)

        if save_table:
            if save_table_name is None:
                save_table_name = self.metric_label + '_rnn_prediction.csv'

            predicted.to_csv(save_table_name)

        if save_plot:
            if save_plot_name is None:
                save_plot_name = self.metric_label + '_rnn_predication.png'

            fig = ax.get_figure()
            fig.savefig(save_plot_name)



    def _rnn_train_df(self, data, epoch):

        # build up model
        model = Sequential()
        model.add(LSTM(self.HIDDEN_NEURONS, input_dim=self.IN_OUT_NEURONS, return_sequences=False))
        model.add(Dense(self.IN_OUT_NEURONS, input_dim=self.HIDDEN_NEURONS))
        model.add(Activation("linear"))

        model.compile(loss="mean_squared_error", optimizer="rmsprop")

        # generate data
        (X_train, y_train), (X_test, y_test) = self.train_test_split(data)  # retrieve data
        print('==========X_train============')
        print(X_test)
        model.fit(X_train, y_train, batch_size=450, nb_epoch=epoch, validation_split=0.05)

        predicted = model.predict(X_test)
        rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

        # and maybe plot it
        # pd.DataFrame(predicted).to_csv("predicted.csv").plot()
        # pd.DataFrame(y_test).to_csv("test_data.csv").polt()

        return predicted, y_test, rmse

    def _calculate_scale_exponent(self, df):
        f = lambda x: len(decimal.Decimal(x).as_tuple().digits) + decimal.Decimal(x).as_tuple().exponent

        temp_df = df.applymap(f)
        scale_exponent = int(temp_df.mean())
        return scale_exponent

    def _load_data(self, data, n_prev=100):
        """
        data should be pd.DataFrame()
        """

        docX, docY = [], []
        for i in range(len(data) - n_prev):
            docX.append(data.iloc[i:i + n_prev].as_matrix())
            docY.append(data.iloc[i + n_prev].as_matrix())
        alsX = np.array(docX)
        alsY = np.array(docY)

        return alsX, alsY

    def train_test_split(self, df, test_size=0.1):
        """
        This just splits data to training and testing parts
        """
        ntrn = round(len(df) * (1 - test_size))

        X_train, y_train = self._load_data(df.iloc[0:ntrn])
        X_test, y_test = self._load_data(df.iloc[ntrn:])

        return (X_train, y_train), (X_test, y_test)

    # def _scale_up_df(self, df, power_val):
    #     scale_up_val = pow(0.1, power_val)
    #     df = df * scale_up_val
    #     return df
    #
    # def _scale_down_df(self, df, power_val):
    #     scale_down_val = pow(10, power_val)
    #     df = df * scale_down_val
    #     return df

    def _scale_df_by(self, base_val):
        def _scale_df_wrapper(df):
            exponent_val = self._calculate_scale_exponent(df)
            print("exponent_val")
            print(exponent_val)
            scale_val = pow(base_val, exponent_val)
            print("scale_val")
            print(scale_val)
            df = df * scale_val
            return df
        return _scale_df_wrapper

    class _HelpAction(argparse._HelpAction):

        def __call__(self, parser, namespace, values, option_string=None):
            #parser.print_help()
            print(parser.format_help())
            # retrieve subparsers from parser
            subparsers_actions = [
                action for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)]
            # there will probably only be one subparser_action,
            # but better save than sorry
            for subparsers_action in subparsers_actions:
                # get all subparsers and print help
                for choice, subparser in subparsers_action.choices.items():
                    print("MetricBoy Command '{}'".format(choice))
                    print(subparser.format_help())

            parser.exit()
if __name__ == '__main__':
    MetricBoy()

