# MetricBoy

MetricBoy is a client for querying, plotting and predicating metric data. Running Machine learning(RNN) against your query metric to perform predictation or anomaly detection based on real-time data.


***
## MetricBoy Help Documentation


            usage: MetricBoy <command> [<args>]
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
          

            optional arguments:
              --metric_exp METRIC_EXP
                                    Metric expression
              --table               display metric data table
              --save_plot           save the result as a plot file
              --save_plot_file SAVE_PLOT_FILE
                                    specify file name of saved plot file
              --save_table          save the result as a table file
              --save_table_file SAVE_TABLE_FILE
                                    specify file name of saved table file
              --help, -h            help for help if you need some help

            MetricBoy commands:
              MetricBoy provides the following commands

              {login,metric,dashboard,predicate}
                login               login command for user login
                metric              metric command for query one single metric
                dashboard           dashboard command for query metrics
                predicate           predicate command for predicating metric data

            MetricBoy Command 'login'
            usage: MetricBoy <command> [<args>] login [-h] [--username USERNAME]
                                                      [--password PASSWORD]

            optional arguments:
              -h, --help           show this help message and exit
              --username USERNAME  username
              --password PASSWORD  password

            MetricBoy Command 'metric'
            usage: MetricBoy <command> [<args>] metric [-h]
                                                       [--plot_type {line,bar,barh,hist,box,kde,area,pie,scatter}]

            optional arguments:
              -h, --help            show this help message and exit
              --plot_type {line,bar,barh,hist,box,kde,area,pie,scatter}
                                    chart type for plotting, please choose one from
                                    ['line', 'bar', 'barh', 'hist', 'box', 'kde', 'area',
                                    'pie', 'scatter']

            MetricBoy Command 'dashboard'
            usage: MetricBoy <command> [<args>] dashboard [-h] [--metric_file METRIC_FILE]
                                                          [--merge]

            optional arguments:
              -h, --help            show this help message and exit
              --metric_file METRIC_FILE
                                    input file where metric expressions stored
              --merge               if set True, metrics will be merged in one table/chart

            MetricBoy Command 'predicate'
            usage: MetricBoy <command> [<args>] predicate [-h] --epoch EPOCH

            optional arguments:
              -h, --help     show this help message and exit
              --epoch EPOCH  epoch configuration for recurrent neural network
## Output Screenshot Example:

### 1.metric
***
![argus custom query latency](https://cloud.githubusercontent.com/assets/7905330/24633842/66734b00-187f-11e7-9bd7-31618e44a713.png)
### 2.dashboard
***

![argus core datapoint readsargus core datapoint writes_dashboard](https://cloud.githubusercontent.com/assets/7905330/24633848/7ac7358a-187f-11e7-90c7-d81d56e62ad0.png)
### 3.predictation
***

![predicted_6000dp_1000epoch](https://cloud.githubusercontent.com/assets/7905330/24633878/a810b494-187f-11e7-9dde-8f34e1a45a50.png)
![test_6000dp_1000epoch](https://cloud.githubusercontent.com/assets/7905330/24633883/b259712a-187f-11e7-8f42-6f97ba290952.png)


